"""LiDAR: where the robot is (a position fix for the EKF), a map of the obstacles, and a safety stop.

The robot is told only the walls of the room (the floor plan). It finds the obstacles itself with the LiDAR.

1. Position fix (wall matching, used by the EKF)
   Every scan point is moved into the room with the predicted pose. Points within MATCH_GATE (0.15 m) of a wall are
   taken to be that wall; everything else (obstacles, the charging station) is ignored. The random rooms keep obstacles
   at least 0.40 m from the walls, and the stretch of the east wall around the given station position is skipped.
   For a point p_B (robot frame) on a wall with normal n at distance d:  residual  r = n . (R(theta) p_B + t) - d.
   The pose (x, y, theta) that makes the residuals smallest is found by weighted least squares (Gauss-Newton, a few
   iterations): each row of the Jacobian is J = [n_x, n_y, n . dR/dtheta p_B], weight 1 / sigma^2 with sigma the
   LDS-01 range noise of that beam. The fix z = (x, y, theta) has covariance R = (J^T W J)^-1, scaled by lidar_r_scale^2
   (chosen by experiments/tune_kalman_filter.py from the filter consistency). Two walls that
   are not parallel are needed (a corridor alone leaves the position along it unknown), so the fix is used only when
   at least MIN_WALL_POINTS points lie on an x wall and on a y wall.

2. Occupancy map (log-odds, the standard method)
   Every grid cell (5 cm) holds a number L, the log-odds that it is occupied, starting at 0 (unknown).
   For each beam:  cells the beam passed through:  L = L - 0.4   ("probably free")
                   the cell where the beam hit:     L = L + 0.85  ("probably occupied")
   A cell is an obstacle when L > 0.5, so one hit is enough and later beams passing through can clear it again.
   L is clamped to [-2, 3.5] so the map can still change.
   Only the first 2.5 m of each beam is used: beyond that the LDS-01 error (5 percent of the range, over 12 cm) is
   larger than two map cells, and far wall hits would appear as obstacles inside the room. A longer beam (or one with
   no return) just clears the cells along its first 2.5 m.
   A hit within 0.20 m of a known wall is taken to be that wall, so a small pose error does not draw a "ghost wall"
   inside the room. The charging station (0.08 m deep, against the east wall) falls in this band too.

3. Planning map
   walls + cells the LiDAR has marked as obstacles. Cells not seen yet count as free, so the robot plans
   optimistically and replans when the LiDAR shows something on its path (path_blocked).

4. Safety stop (collision detection)
   If any beam within +-35 degrees of straight ahead ends closer than the robot radius + 0.10 m to the robot centre,
   the robot should stop (front_blocked). The beams are first moved into the robot frame (the LiDAR sits 0.032 m behind
   the centre).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from robot_code import room_map as km
from robot_code.data_types import LidarScan, StateEstimate, wrap
from robot_code.path_planning import Grid, empty_room_grid

L_FREE, L_OCC = -0.4, 0.85        # log-odds change per beam (assumption, the usual textbook values)
L_MIN, L_MAX = -2.0, 3.5
L_OBSTACLE = 0.5                  # a cell is an obstacle above this log-odds
MAP_RANGE = 2.5                   # m, only this much of each beam is used for the map
WALL_BAND = 0.20                  # m, hits this close to a known wall are taken to be that wall
FRONT_HALF_ANGLE = np.deg2rad(35.0)
FRONT_MARGIN = 0.10               # m beyond the robot radius
MATCH_GATE = 0.15                 # m, scan points this close to a wall are matched to it
MIN_WALL_POINTS = 10              # points needed on an x wall and on a y wall
STATION_SKIP = km.STATION_HALF_WIDTH + km.DOCK_HINT_ERROR + 0.05   # m either side of the given station position
LIDAR_SPEC_JSON = Path(__file__).resolve().parents[1] / "settings" / "lidar_spec.json"


# ------------------------------------------------------------------ 1. position fix from the known walls
@dataclass(frozen=True)
class LidarFix:
    """A pose measurement from the LiDAR: z = (x, y, theta) in W with covariance R."""
    t: float
    z: np.ndarray
    R: np.ndarray
    n_points: int


_SPEC = json.loads(LIDAR_SPEC_JSON.read_text())


def range_sigma(r: np.ndarray) -> np.ndarray:
    """LDS-01 range noise (datasheet: 15 mm below 0.5 m, 5 percent of the range beyond), one standard deviation."""
    return np.where(r < _SPEC["near_limit_m"], _SPEC["sigma_near_m"], _SPEC["sigma_far_fraction"] * r)


# The four walls as n . p = d (unit normal n, distance d).
WALLS = ((np.array([1.0, 0.0]), km.ROOM_X_MIN), (np.array([1.0, 0.0]), km.ROOM_X_MAX),
         (np.array([0.0, 1.0]), km.ROOM_Y_MIN), (np.array([0.0, 1.0]), km.ROOM_Y_MAX))


def wall_fix(scan: LidarScan, pose_guess, dock_hint_y: float, lidar_offset_xy=(-0.032, 0.0),
             r_scale: float = 1.0, iterations: int = 4) -> Optional[LidarFix]:
    """Robot pose from one scan matched to the four known walls (weighted least squares, see the module docstring).
    pose_guess is the EKF prediction. Returns None when the walls seen do not fix the whole pose."""
    r = np.asarray(scan.ranges, dtype=float)
    a = np.asarray(scan.angles, dtype=float)
    ok = np.isfinite(r)
    r, a = r[ok], a[ok]
    pB = np.column_stack([lidar_offset_xy[0] + r * np.cos(a), lidar_offset_xy[1] + r * np.sin(a)])
    sig_range = range_sigma(r)
    N = np.array([n for n, _ in WALLS])
    D = np.array([d for _, d in WALLS])
    x = np.array(pose_guess, dtype=float)
    for _ in range(iterations):
        c, s = np.cos(x[2]), np.sin(x[2])
        P = np.column_stack([x[0] + c * pB[:, 0] - s * pB[:, 1], x[1] + s * pB[:, 0] + c * pB[:, 1]])
        res_all = P @ N.T - D                              # residual of every point to every wall
        w = np.argmin(np.abs(res_all), axis=1)             # nearest wall
        res = res_all[np.arange(len(P)), w]
        use = np.abs(res) < MATCH_GATE
        use &= ~((w == 1) & (np.abs(P[:, 1] - dock_hint_y) < STATION_SKIP))   # skip the station on the east wall
        n_x, n_y = int(np.sum(use & (w < 2))), int(np.sum(use & (w >= 2)))
        if n_x < MIN_WALL_POINTS or n_y < MIN_WALL_POINTS:
            return None
        n = N[w[use]]
        dP = np.column_stack([-s * pB[use, 0] - c * pB[use, 1], c * pB[use, 0] - s * pB[use, 1]])
        J = np.column_stack([n, np.sum(n * dP, axis=1)])
        W = 1.0 / sig_range[use] ** 2
        A = J.T @ (W[:, None] * J)
        dx = -np.linalg.solve(A, J.T @ (W * res[use]))
        x = x + dx
        x[2] = float(wrap(x[2]))
        if np.abs(dx).max() < 1e-5:
            break
    return LidarFix(scan.t, x, np.linalg.inv(A) * r_scale ** 2, int(use.sum()))


class LidarMap:
    def __init__(self, lidar_offset_xy=(-0.032, 0.0), r_max: float = 3.5, res: float = 0.05):
        self.offset = np.asarray(lidar_offset_xy, dtype=float)   # LiDAR position in the robot frame
        self.r_max = r_max
        self.walls = empty_room_grid(res)                        # cells outside the room are walls
        self.wall_distance = self.walls.clearance()               # distance of each cell to the nearest known wall
        self.logodds = np.zeros(self.walls.shape)
        self.n_scans = 0

    # ------------------------------------------------------------------ 2. update the map with one scan
    def add_scan(self, scan: LidarScan, est: StateEstimate) -> None:
        x, y, th = est.x_hat
        c, s = np.cos(th), np.sin(th)
        ox, oy = x + c * self.offset[0] - s * self.offset[1], y + s * self.offset[0] + c * self.offset[1]
        ranges = np.asarray(scan.ranges, dtype=float)
        hit = np.isfinite(ranges) & (ranges <= MAP_RANGE)
        dist = np.where(hit, ranges, MAP_RANGE)
        ang = th + np.asarray(scan.angles, dtype=float)
        g = self.walls
        # Points every half cell along each beam, up to (not including) the end point: these cells are free.
        step = 0.5 * g.res
        n = int(np.ceil(MAP_RANGE / step))
        t = np.arange(n) * step
        along = t[None, :] < np.where(hit, dist - g.res, dist)[:, None]
        px = ox + np.cos(ang)[:, None] * t[None, :]
        py = oy + np.sin(ang)[:, None] * t[None, :]
        self._add(px[along], py[along], L_FREE)
        # The cell where each beam hit: occupied, unless it is the known wall seen with a small pose error.
        hx, hy = ox + np.cos(ang[hit]) * dist[hit], oy + np.sin(ang[hit]) * dist[hit]
        i, j = g.to_cell_array(hx, hy)
        near_wall = np.ones(len(hx), bool)
        inside = (i >= 0) & (i < g.shape[0]) & (j >= 0) & (j < g.shape[1])
        near_wall[inside] = self.wall_distance[i[inside], j[inside]] < WALL_BAND
        self._add(hx[~near_wall], hy[~near_wall], L_OCC)
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)
        self.n_scans += 1

    def _add(self, px, py, value) -> None:
        g = self.walls
        i, j = g.to_cell_array(px, py)
        ok = (i >= 0) & (i < g.shape[0]) & (j >= 0) & (j < g.shape[1])
        cells = np.unique(np.stack([i[ok], j[ok]], axis=1), axis=0)   # each cell once per scan
        if len(cells):
            self.logodds[cells[:, 0], cells[:, 1]] += value

    # ------------------------------------------------------------------ 3. the map used for planning
    def obstacles(self) -> np.ndarray:
        return self.logodds > L_OBSTACLE

    def grid(self) -> Grid:
        g = self.walls
        return Grid(g.x0, g.y0, g.res, g.occ | self.obstacles())

    def path_blocked(self, xs, ys, clearance_min: float) -> bool:
        """True if any of the path points (the rest of the current path) is closer than clearance_min to a wall or
        a discovered obstacle."""
        g = self.grid()
        clr = g.clearance()
        for x, y in zip(xs, ys):
            i, j = g.to_cell(x, y)
            if g.inside(i, j) and clr[i, j] < clearance_min:
                return True
        return False


# ------------------------------------------------------------------ 4. safety stop
def front_distance(scan: LidarScan, lidar_offset_xy=(-0.032, 0.0)) -> float:
    """Distance from the robot centre to the nearest LiDAR return within +-35 degrees of straight ahead."""
    r = np.asarray(scan.ranges, dtype=float)
    a = np.asarray(scan.angles, dtype=float)
    ok = np.isfinite(r)
    px = lidar_offset_xy[0] + r[ok] * np.cos(a[ok])     # beam end points in the robot frame
    py = lidar_offset_xy[1] + r[ok] * np.sin(a[ok])
    ahead = (px > 0) & (np.abs(np.arctan2(py, px)) <= FRONT_HALF_ANGLE)
    return float(np.hypot(px[ahead], py[ahead]).min()) if ahead.any() else np.inf


def front_blocked(scan: LidarScan, lidar_offset_xy=(-0.032, 0.0)) -> bool:
    return front_distance(scan, lidar_offset_xy) < km.ROBOT_RADIUS + FRONT_MARGIN

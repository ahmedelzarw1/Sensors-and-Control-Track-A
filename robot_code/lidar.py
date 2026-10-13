"""LiDAR: a map of the obstacles built from the scans, and a safety stop.

The robot is told only the walls and the charging station (the floor plan). It finds the obstacles itself with the
LiDAR. Every scan is placed in the room with the EKF ESTIMATE of the robot's pose, never the true position.

1. Occupancy map (log-odds, the standard method)
   Every grid cell (5 cm) holds a number L, the log-odds that it is occupied, starting at 0 (unknown).
   For each beam:  cells the beam passed through:  L = L - 0.4   ("probably free")
                   the cell where the beam hit:     L = L + 0.85  ("probably occupied")
   A cell is an obstacle when L > 0.5, so one hit is enough and later beams passing through can clear it again.
   L is clamped to [-2, 3.5] so the map can still change.
   Only the first 2.5 m of each beam is used: beyond that the LDS-01 error (5 percent of the range, over 12 cm) is
   larger than two map cells, and far wall hits would appear as obstacles inside the room. A longer beam (or one with
   no return) just clears the cells along its first 2.5 m.
   A hit within 0.20 m of a known wall or the charging station is taken to be that wall: when the pose estimate is
   off (for example 0.14 m before the first marker fixes), the walls would otherwise be drawn as a "ghost wall" inside
   the room and could close a gap. The random rooms keep obstacles at least 0.40 m from the walls.

2. Planning map
   walls + charging station + cells the LiDAR has marked as obstacles. Cells not seen yet count as free, so the robot
   plans optimistically and replans when the LiDAR shows something on its path (path_blocked).

3. Safety stop
   If any beam within +-35 degrees of straight ahead ends closer than the robot radius + 0.10 m to the robot centre,
   the robot should stop (front_blocked). The beams are first moved into the robot frame (the LiDAR sits 0.032 m behind
   the centre).
"""
from __future__ import annotations

import numpy as np

from robot_code import room_map as km
from robot_code.data_types import LidarScan, StateEstimate
from robot_code.path_planning import Grid, empty_room_grid

L_FREE, L_OCC = -0.4, 0.85        # log-odds change per beam (assumption, the usual textbook values)
L_MIN, L_MAX = -2.0, 3.5
L_OBSTACLE = 0.5                  # a cell is an obstacle above this log-odds
MAP_RANGE = 2.5                   # m, only this much of each beam is used for the map
WALL_BAND = 0.20                  # m, hits this close to a known wall are taken to be that wall
FRONT_HALF_ANGLE = np.deg2rad(35.0)
FRONT_MARGIN = 0.10               # m beyond the robot radius


class LidarMap:
    def __init__(self, lidar_offset_xy=(-0.032, 0.0), r_max: float = 3.5, res: float = 0.05):
        self.offset = np.asarray(lidar_offset_xy, dtype=float)   # LiDAR position in the robot frame
        self.r_max = r_max
        self.walls = empty_room_grid(res)                        # cells outside the room are walls
        X, Y = self.walls.centres()
        self.walls.occ |= km.footprint_contains(km.station_footprint(), X, Y)
        self.wall_distance = self.walls.clearance()               # distance of each cell to the nearest known wall
        self.logodds = np.zeros(self.walls.shape)
        self.n_scans = 0

    # ------------------------------------------------------------------ 1. update the map with one scan
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

    # ------------------------------------------------------------------ 2. the map used for planning
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


# ------------------------------------------------------------------ 3. safety stop
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

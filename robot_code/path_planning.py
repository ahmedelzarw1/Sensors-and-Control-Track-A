"""Path planning: a path around the obstacles that the robot can actually drive, as a time-indexed reference.

Four parts, in the order they are used:
1. Grid and A*: occupancy grid of the map, obstacle inflation, A* search.
2. Reference: integrate_reference builds the reference with the unicycle model so pose and speeds agree.
3. Smooth path: shortcutting, circular-arc corners, speed profile, time sampling.
4. Plan: plan_reference, from the start pose to a goal pose.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt
from robot_code import room_map as km
from robot_code.unicycle_model import f_unicycle
from robot_code.data_types import ReferenceTrajectory
from robot_code.data_types import ReferenceTrajectory, wrap
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================================================
# 1. GRID AND A*
# Occupancy grid, clearance map and A* search.
#
# The same grid type is filled in two ways:
# - from the known map (robot_code/room_map.py), the Stage 1 planner and the fallback, and
# - from LiDAR scans placed with the ESTIMATED pose (Stage 2, still to be written).
#
# Obstacle inflation uses the Euclidean distance transform: every cell stores its distance to the nearest occupied cell
# (the clearance). A cell is traversable if its clearance is at least the robot radius plus a safety margin, which is
# the same as inflating every obstacle by that amount. A* then searches the traversable cells (8-connected) with a small
# extra cost near obstacles so the path keeps away from them when there is room.
# ==============================================================================================================

@dataclass
class Grid:
    x0: float          # world x of the centre of cell (0, 0)
    y0: float
    res: float         # cell size, m
    occ: np.ndarray    # bool (ny, nx), True = occupied

    @property
    def shape(self):
        return self.occ.shape

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((y - self.y0) / self.res)), int(round((x - self.x0) / self.res))

    def to_cell_array(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """to_cell for arrays of points (rows i, columns j)."""
        return (np.round((np.asarray(y) - self.y0) / self.res).astype(int),
                np.round((np.asarray(x) - self.x0) / self.res).astype(int))

    def to_world(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + j * self.res, self.y0 + i * self.res

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        ii, jj = np.mgrid[0:self.occ.shape[0], 0:self.occ.shape[1]]
        return self.x0 + jj * self.res, self.y0 + ii * self.res

    def clearance(self) -> np.ndarray:
        """Distance from each cell centre to the nearest occupied cell centre, m."""
        return distance_transform_edt(~self.occ) * self.res


def empty_room_grid(res: float = 0.05, pad: float = 0.0) -> Grid:
    """Grid covering the room interior. Cells outside the room count as occupied (the walls)."""
    x0, y0 = km.ROOM_X_MIN - pad, km.ROOM_Y_MIN - pad
    nx = int(round((km.ROOM_X_MAX - km.ROOM_X_MIN + 2 * pad) / res)) + 1
    ny = int(round((km.ROOM_Y_MAX - km.ROOM_Y_MIN + 2 * pad) / res)) + 1
    g = Grid(x0, y0, res, np.zeros((ny, nx), bool))
    X, Y = g.centres()
    g.occ |= (X <= km.ROOM_X_MIN) | (X >= km.ROOM_X_MAX) | (Y <= km.ROOM_Y_MIN) | (Y >= km.ROOM_Y_MAX)
    return g


def known_map_grid(res: float = 0.05, layout: "km.Layout | None" = None) -> Grid:
    """Rasterise the known map: walls, obstacles (of the given layout, default the fixed room) and the station."""
    g = empty_room_grid(res)
    X, Y = g.centres()
    for ob in km.obstacle_footprints(layout):
        g.occ |= km.footprint_contains(ob, X, Y)
    return g


_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def astar(grid: Grid, start_xy, goal_xy, clearance_min: float, clearance_pref: float = 0.35,
          near_cost: float = 2.0) -> list[tuple[float, float]] | None:
    """Shortest 8-connected path between two world points over cells with clearance >= clearance_min.

    Cost per step is the step length times (1 + near_cost * max(0, clearance_pref - c) / clearance_pref), so cells
    closer than clearance_pref to an obstacle are discouraged but allowed. Returns world points or None.
    """
    clr = grid.clearance()
    free = clr >= clearance_min
    s, g = grid.to_cell(*start_xy), grid.to_cell(*goal_xy)
    # The start may lie inside the inflated zone (the robot itself is never inside an obstacle). Allow leaving it.
    if not grid.inside(*s) or not grid.inside(*g) or not free[g]:
        return None
    penalty = 1.0 + near_cost * np.clip(clearance_pref - clr, 0.0, None) / clearance_pref
    h = lambda c: grid.res * np.hypot(c[0] - g[0], c[1] - g[1])
    open_ = [(h(s), 0.0, s)]
    came, cost = {s: None}, {s: 0.0}
    while open_:
        _, gc, cur = heapq.heappop(open_)
        if cur == g:
            break
        if gc > cost[cur]:
            continue
        for di, dj in _NEIGHBOURS:
            nb = (cur[0] + di, cur[1] + dj)
            if not grid.inside(*nb):
                continue
            if not free[nb] and not (not free[cur] and clr[nb] >= clr[cur]):
                continue  # blocked, unless we are still escaping the inflated zone around the start
            step = grid.res * (1.4142135623730951 if di and dj else 1.0) * penalty[nb]
            nc = gc + step
            if nc < cost.get(nb, np.inf):
                cost[nb], came[nb] = nc, cur
                heapq.heappush(open_, (nc + h(nb), nc, nb))
    if g not in came:
        return None
    cells, c = [], g
    while c is not None:
        cells.append(c)
        c = came[c]
    cells.reverse()
    pts = [grid.to_world(*c) for c in cells]
    pts[0], pts[-1] = tuple(start_xy), tuple(goal_xy)
    return pts


def segment_clear(clr: np.ndarray, grid: Grid, a, b, clearance_min: float) -> bool:
    """True if every grid sample along the segment a-b has clearance >= clearance_min."""
    n = max(2, int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * grid.res))) + 1)
    for t in np.linspace(0.0, 1.0, n):
        i, j = grid.to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if not grid.inside(i, j) or clr[i, j] < clearance_min:
            return False
    return True


# ==============================================================================================================
# 2. REFERENCE
# Build a ReferenceTrajectory from a velocity profile.
#
# The docking trajectory (vision part) and the obstacle-aware planner (planning part) both
# end up as time-indexed (v, omega) profiles. This integrates the exact unicycle so that the pose
# and velocity columns are consistent, which the feedforward term of the tracker relies on.
# ==============================================================================================================

def integrate_reference(pose0, v, omega, Ts: float) -> ReferenceTrajectory:
    """Reference with N samples from N velocity pairs. Sample k is the pose at time k*Ts."""
    v, omega = np.asarray(v, dtype=float), np.asarray(omega, dtype=float)
    N = len(v)
    X = np.zeros((N, 3))
    X[0] = pose0
    for k in range(N - 1):
        X[k + 1] = f_unicycle(X[k], (v[k], omega[k]), Ts)
    return ReferenceTrajectory(t=Ts * np.arange(N), x=X[:, 0], y=X[:, 1], theta=X[:, 2], v=v, omega=omega)


# ==============================================================================================================
# 3. SMOOTH PATH
# Path smoothing and the velocity profile that turns a geometric path into a ReferenceTrajectory.
#
# Steps
# 1. Shortcut: from the A* cell path keep only the waypoints needed so every straight segment stays clear.
# 2. Fillet: replace each corner by a circular arc tangent to both segments. The radius is the largest value up to
#    r_max that fits the segment lengths and keeps the arc clear. Lines and arcs are exactly drivable by a unicycle
#    (constant omega / v on each piece), so the reference pose and velocity columns agree.
# 3. Dense samples of (s, x, y, theta, curvature) along the path.
# 4. Velocity profile: speed limited by the cruise speed, by omega_max / |curvature| on arcs, and by an acceleration
#    limit (forward and backward passes over s). Start and end speeds are inputs, and the end speed is kept above zero
#    because the tracking model is not controllable at zero reference speed (robot_code/unicycle_model.py, ctrb).
# 5. Time sampling at Ts. Sample k uses v_k = (s_{k+1} - s_k) / Ts and omega_k = (theta_{k+1} - theta_k) / Ts, so the
#    integrated reference (robot_code/path_planning.py) reproduces the path heading exactly.
# ==============================================================================================================

@dataclass(frozen=True)
class ProfileLimits:
    """Reference limits. Values are design choices (assumptions), below the tracker limits of 0.22 m/s and 2.84 rad/s
    so the feedback has headroom."""
    v_cruise: float = 0.15     # m/s
    omega_max: float = 0.8     # rad/s on arcs
    a_max: float = 0.15        # m/s^2
    v_min: float = 0.03        # m/s, lowest reference speed (keeps the tracking model controllable)


@dataclass
class DensePath:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray

    @property
    def length(self) -> float:
        return float(self.s[-1])


def shortcut(points, grid: Grid, clearance_min: float) -> list:
    """Greedy line-of-sight shortcutting of a polyline."""
    clr = grid.clearance()
    pts = [np.asarray(p, float) for p in points]
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not segment_clear(clr, grid, pts[i], pts[j], clearance_min):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _arc_points(p_prev, p, p_next, r):
    """Tangent points, centre and signed turn angle of the fillet of radius r at corner p."""
    d1 = (p - p_prev) / np.linalg.norm(p - p_prev)
    d2 = (p_next - p) / np.linalg.norm(p_next - p)
    turn = float(np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2))
    tlen = r * np.tan(abs(turn) / 2)
    a, b = p - d1 * tlen, p + d2 * tlen
    nrm = np.array([-d1[1], d1[0]]) * np.sign(turn)  # towards the centre
    return a, b, a + nrm * r, turn, tlen


def fillet_path(waypoints, grid: Grid, clearance_min: float, r_max: float = 0.35, ds: float = 0.01,
                theta_start: float | None = None, theta_end: float | None = None) -> DensePath:
    """Line and arc path through the waypoints, densely sampled. The corners are rounded with the largest feasible
    radius up to r_max whose arc keeps clearance_min (shrinking the radius until it fits)."""
    W = [np.asarray(p, float) for p in waypoints]
    clr = grid.clearance()
    seg_len = [np.linalg.norm(W[i + 1] - W[i]) for i in range(len(W) - 1)]
    # Each segment can give at most half its length to each of its two corners (all of it at the path ends).
    avail = []
    for i in range(1, len(W) - 1):
        a = seg_len[i - 1] if i == 1 else 0.5 * seg_len[i - 1]
        b = seg_len[i] if i == len(W) - 2 else 0.5 * seg_len[i]
        avail.append(min(a, b))
    arcs = []
    for k, i in enumerate(range(1, len(W) - 1)):
        _, _, _, turn, _ = _arc_points(W[i - 1], W[i], W[i + 1], 1.0)
        if abs(turn) < 1e-6:
            arcs.append(None)
            continue
        r = min(r_max, 0.999 * avail[k] / np.tan(abs(turn) / 2))
        while r > 0.02:
            a, b, c, turn, _ = _arc_points(W[i - 1], W[i], W[i + 1], r)
            if _arc_clear(clr, grid, a, c, turn, r, clearance_min):
                break
            r *= 0.8
        arcs.append(_arc_points(W[i - 1], W[i], W[i + 1], max(r, 0.02)))
    # Assemble straight pieces and arcs into dense samples.
    xs, ys, ths, ks = [], [], [], []
    cur = W[0]
    for k in range(len(W) - 1):
        nxt_start = arcs[k][0] if k < len(arcs) and arcs[k] is not None else W[k + 1]
        _line(cur, nxt_start, ds, xs, ys, ths, ks)
        if k < len(arcs) and arcs[k] is not None:
            a, b, c, turn, _ = arcs[k]
            r = np.linalg.norm(a - c)
            _arc(a, c, turn, r, ds, xs, ys, ths, ks)
            cur = b
        else:
            cur = W[k + 1]
    xs.append(cur[0]); ys.append(cur[1]); ths.append(ths[-1] if ths else 0.0); ks.append(0.0)
    x, y = np.array(xs), np.array(ys)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return DensePath(s, x, y, np.unwrap(np.array(ths)), np.array(ks))


def _line(a, b, ds, xs, ys, ths, ks):
    L = float(np.linalg.norm(b - a))
    if L < 1e-9:
        return
    th = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
    n = max(1, int(np.ceil(L / ds)))
    for t in np.arange(n) / n:
        xs.append(a[0] + t * (b[0] - a[0])); ys.append(a[1] + t * (b[1] - a[1])); ths.append(th); ks.append(0.0)


def _arc(a, c, turn, r, ds, xs, ys, ths, ks):
    phi0 = float(np.arctan2(a[1] - c[1], a[0] - c[0]))
    n = max(1, int(np.ceil(abs(turn) * r / ds)))
    sgn = np.sign(turn)
    for t in np.arange(n) / n:
        phi = phi0 + t * turn
        xs.append(c[0] + r * np.cos(phi)); ys.append(c[1] + r * np.sin(phi))
        ths.append(phi + sgn * np.pi / 2); ks.append(sgn / r)


def _arc_clear(clr, grid, a, c, turn, r, clearance_min) -> bool:
    phi0 = float(np.arctan2(a[1] - c[1], a[0] - c[0]))
    for t in np.linspace(0, 1, max(3, int(abs(turn) * r / (0.5 * grid.res)))):
        p = c + r * np.array([np.cos(phi0 + t * turn), np.sin(phi0 + t * turn)])
        i, j = grid.to_cell(*p)
        if not grid.inside(i, j) or clr[i, j] < clearance_min:
            return False
    return True


def speed_profile(path: DensePath, lim: ProfileLimits, v_start: float, v_end: float) -> np.ndarray:
    """Speed at each path sample: curvature and cruise limits, then acceleration-limited forward/backward passes."""
    # The curvature limit is applied after the minimum speed, so the reference turn rate v * kappa never exceeds
    # omega_max. On a tight corner the speed then drops below v_min, but stays positive (the LQR design speed floor
    # in robot_code/lqr_controller.py covers it). Applying v_min last made sharp corners ask for about 1.5 rad/s, and the fast
    # turn reversals that followed are the hardest manoeuvre for the odometry.
    v_curv = lim.omega_max / np.maximum(np.abs(path.kappa), 1e-9)
    v = np.full_like(path.s, lim.v_cruise)
    v[0], v[-1] = max(v_start, lim.v_min), max(v_end, lim.v_min)
    v = np.minimum(np.maximum(v, lim.v_min), v_curv)
    ds = np.diff(path.s)
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * lim.a_max * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * lim.a_max * ds[i]))
    return v


def reference_from_path(path: DensePath, v_profile: np.ndarray, Ts: float, extend: float = 0.0,
                        v_extend: float | None = None) -> ReferenceTrajectory:
    """Time-sample a path with its speed profile. If extend > 0 the reference continues straight past the end of the
    path for that distance at v_extend (used by docking, whose stop rule ends the motion before the reference does)."""
    s, th = path.s, path.theta
    if extend > 0:
        n = int(np.ceil(extend / 0.01))
        s = np.concatenate([s, s[-1] + 0.01 * np.arange(1, n + 1)])
        th = np.concatenate([th, np.full(n, th[-1])])
        v_profile = np.concatenate([v_profile, np.full(n, v_profile[-1] if v_extend is None else v_extend)])
    # Time at each sample: t = integral of ds / v (trapezoidal on 1/v).
    inv = 1.0 / np.maximum(v_profile, 1e-3)
    t = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(s))])
    tk = np.arange(0.0, t[-1] + 1e-9, Ts)
    sk = np.interp(tk, t, s)
    thk = np.interp(sk, s, th)
    v = np.diff(sk) / Ts
    w = np.diff(thk) / Ts
    v, w = np.append(v, v[-1] if len(v) else v_profile[-1]), np.append(w, 0.0)
    return integrate_reference([path.x[0], path.y[0], float(wrap(th[0]))], v, w, Ts)


# ==============================================================================================================
# 4. PLAN
# Obstacle-aware path from the estimated pose to a goal pose, as a ReferenceTrajectory for the LQR tracker.
#
# plan_reference(grid, start_pose, goal_pose) runs A* to an approach point behind the goal (so the path arrives with
# the goal heading), appends the goal, shortcuts, fillets the corners and time-samples with the speed profile.
# The grid may come from the known map (Stage 1) or from the LiDAR occupancy map built with the estimated pose
# (Stage 2). The planner never sees ground truth: its inputs are the estimate and a grid.
# ==============================================================================================================

@dataclass(frozen=True)
class PlannerParams:
    # Clearance from obstacle cells required for the robot centre: robot radius plus a safety margin (assumption)
    # covering tracking error and estimation error.
    margin: float = 0.075
    arc_margin: float = 0.045        # smaller margin accepted on the rounded corners
    approach: float = 0.40           # m, straight run into the goal along the goal heading
    departure: float = 0.25          # m, straight run out of the start along the current heading
    departure_max_turn: float = np.deg2rad(100.0)  # larger first turns are made in place instead
    r_max: float = 0.35              # m, largest corner radius
    limits: ProfileLimits = field(default_factory=ProfileLimits)

    @property
    def clearance(self) -> float:
        return km.ROBOT_RADIUS + self.margin


@dataclass
class Plan:
    waypoints: list
    path: DensePath
    ref: ReferenceTrajectory


def plan_reference(grid: Grid, start_pose, goal_pose, Ts: float, params: PlannerParams = PlannerParams(),
                   v_start: float = 0.0, v_end: float = 0.05) -> Optional[Plan]:
    gx, gy, gth = goal_pose
    app = np.array([gx - params.approach * np.cos(gth), gy - params.approach * np.sin(gth)])
    pts = astar(grid, start_pose[:2], app, params.clearance)
    if pts is None:
        return None
    clr = grid.clearance()
    if not segment_clear(clr, grid, app, np.array([gx, gy]), km.ROBOT_RADIUS + params.arc_margin):
        return None
    wps = shortcut(pts, grid, params.clearance)
    # Departure: if the first leg is within departure_max_turn of the current heading, start with a short straight
    # run along the heading so the first turn is a driven arc. Turning in place makes the largest odometry heading
    # errors on this platform (turn in place 0.98 of the odometry value against 0.99 on arcs, README.md, Robot model), so
    # it is kept for large first turns only.
    p0 = np.asarray(start_pose[:2], float)
    h0 = np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
    d0 = wps[1] - wps[0]
    turn = abs(float(np.arctan2(h0[0] * d0[1] - h0[1] * d0[0], h0 @ d0)))
    dep = p0 + params.departure * h0
    if turn <= params.departure_max_turn and segment_clear(clr, grid, p0, dep, km.ROBOT_RADIUS + params.arc_margin):
        pts2 = astar(grid, dep, app, params.clearance)
        if pts2 is not None:
            wps = [p0] + shortcut(pts2, grid, params.clearance)
    wps = _drop_short(wps + [np.array([gx, gy])])
    path = fillet_path(wps, grid, km.ROBOT_RADIUS + params.arc_margin, r_max=params.r_max)
    v = speed_profile(path, params.limits, v_start, v_end)
    ref = reference_from_path(path, v, Ts)
    return Plan(wps, path, ref)


def _drop_short(wps, min_len: float = 0.02):
    out = [np.asarray(wps[0], float)]
    for p in wps[1:]:
        p = np.asarray(p, float)
        if np.linalg.norm(p - out[-1]) > min_len:
            out.append(p)
        else:
            out[-1] = p  # keep the later point (the goal must stay exact)
    return out

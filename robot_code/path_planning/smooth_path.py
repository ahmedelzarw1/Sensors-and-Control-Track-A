"""Path smoothing and the velocity profile that turns a geometric path into a ReferenceTrajectory.

Steps
1. Shortcut: from the A* cell path keep only the waypoints needed so every straight segment stays clear.
2. Fillet: replace each corner by a circular arc tangent to both segments. The radius is the largest value up to
   r_max that fits the segment lengths and keeps the arc clear. Lines and arcs are exactly drivable by a unicycle
   (constant omega / v on each piece), so the reference pose and velocity columns agree.
3. Dense samples of (s, x, y, theta, curvature) along the path.
4. Velocity profile: speed limited by the cruise speed, by omega_max / |curvature| on arcs, and by an acceleration
   limit (forward and backward passes over s). Start and end speeds are inputs, and the end speed is kept above zero
   because the tracking model is not controllable at zero reference speed (robot_code/movement_maths/unicycle_model.py, ctrb).
5. Time sampling at Ts. Sample k uses v_k = (s_{k+1} - s_k) / Ts and omega_k = (theta_{k+1} - theta_k) / Ts, so the
   integrated reference (robot_code/path_planning/reference_path.py) reproduces the path heading exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_code.path_planning.astar_grid import Grid, segment_clear
from robot_code.path_planning.reference_path import integrate_reference
from robot_code.data_types import ReferenceTrajectory, wrap


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
    # in robot_code/controller/lqr_controller.py covers it). Applying v_min last made sharp corners ask for about 1.5 rad/s, and the fast
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

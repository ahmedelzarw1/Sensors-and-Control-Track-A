"""Plots and reports for a run: the room drawing used everywhere, the per-run CSV log, metrics and figures.

1. Draw room: walls, obstacles, charging station, marker, robot triangles, covariance ellipses.
2. Run report: CSV log, metrics (docking error, tracking error, NEES, NIS) and the plots of each run.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
from robot_code import room_map as km
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from experiments.scores import nees
from robot_code.lqr_controller import Limits
from robot_code.data_types import wrap


# ==============================================================================================================
# 1. DRAW ROOM
# Plot helpers shared by the demo, the trials and the documentation figures.
# ==============================================================================================================

def draw_map(ax, show_dock: bool = True, layout=None) -> None:
    """Room walls, obstacles, charging station and marker (TRUE position), the given (approximate) station position,
    the start point and the end point of the trip. layout None: the fixed demo room."""
    layout = layout if layout is not None else km.default_layout()
    ax.add_patch(mpatches.Rectangle((km.ROOM_X_MIN, km.ROOM_Y_MIN), km.ROOM_X_MAX - km.ROOM_X_MIN,
                                    km.ROOM_Y_MAX - km.ROOM_Y_MIN, fill=False, lw=2.0, ec="0.2"))
    for ob in km.obstacle_footprints(layout):
        if isinstance(ob, km.Box):
            fc = "0.25" if ob.name == "charging_station" else "#d9875a"
            c, s = np.cos(ob.yaw), np.sin(ob.yaw)
            corners = np.array([[-ob.hx, -ob.hy], [ob.hx, -ob.hy], [ob.hx, ob.hy], [-ob.hx, ob.hy]])
            pts = corners @ np.array([[c, s], [-s, c]]) + [ob.x, ob.y]
            ax.add_patch(mpatches.Polygon(pts, closed=True, fc=fc, ec="none"))
        else:
            ax.add_patch(mpatches.Circle((ob.x, ob.y), ob.radius, fc="#5a8cd9", ec="none"))
    mx, my, myaw = km.marker_pose(layout.dock_y)
    half = km.MARKER_SIZE / 2
    ax.plot([mx - half * np.sin(myaw), mx + half * np.sin(myaw)], [my + half * np.cos(myaw), my - half * np.cos(myaw)],
            color="k", lw=4, solid_capstyle="butt")
    if show_dock:
        # The station position the robot was given, and the uncertainty band around it.
        hy = layout.dock_hint_y
        ax.add_patch(mpatches.Rectangle((km.STATION_FRONT_X - 0.03, hy - km.DOCK_HINT_ERROR), 0.03,
                                        2 * km.DOCK_HINT_ERROR, fc="tab:purple", alpha=0.25, ec="none"))
        ax.plot([km.STATION_FRONT_X - 0.05, km.ROOM_X_MAX], [hy, hy], color="tab:purple", lw=1.2)
        end = km.approach_pose(hy)
        ax.plot(*end[:2], "x", color="tab:purple", ms=8)
        ax.annotate("end point", end[:2], textcoords="offset points", xytext=(-22, 8), fontsize=7, color="tab:purple")
        dock = km.dock_pose(layout.dock_y)
        ax.plot(*dock[:2], "k+", ms=8)
        ax.annotate("dock", dock[:2], textcoords="offset points", xytext=(-10, 8), fontsize=7)
        ax.plot(*layout.start[:2], "o", mfc="none", mec="tab:blue", ms=8)
        ax.annotate("start", layout.start[:2], textcoords="offset points", xytext=(-10, 8), fontsize=7, color="tab:blue")
    ax.set_xlim(km.ROOM_X_MIN - 0.1, km.ROOM_X_MAX + 0.1)
    ax.set_ylim(km.ROOM_Y_MIN - 0.1, km.ROOM_Y_MAX + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def lidar_map_rgba(lidar_map) -> tuple[np.ndarray, list]:
    """The obstacle cells the LiDAR has found, as a red RGBA image, and its extent for imshow (origin='lower')."""
    g = lidar_map.walls
    obs = lidar_map.obstacles()
    img = np.zeros(obs.shape + (4,))
    img[obs] = (0.85, 0.1, 0.1, 0.65)
    ny, nx = obs.shape
    extent = [g.x0 - g.res / 2, g.x0 + (nx - 0.5) * g.res, g.y0 - g.res / 2, g.y0 + (ny - 0.5) * g.res]
    return img, extent


def robot_triangle(ax, pose, color, size: float = 0.12, **kw):
    x, y, th = pose
    pts = np.array([[size, 0.0], [-0.6 * size, 0.5 * size], [-0.6 * size, -0.5 * size]])
    c, s = np.cos(th), np.sin(th)
    P = pts @ np.array([[c, s], [-s, c]]) + [x, y]
    return ax.add_patch(mpatches.Polygon(P, closed=True, fc=color, ec="k", lw=0.5, **kw))


def ellipse_points(mean_xy, P_xy, n_sigma: float = 3.0, n: int = 60) -> np.ndarray:
    """Points of the n-sigma covariance ellipse of a 2D position."""
    w, V = np.linalg.eigh(0.5 * (P_xy + P_xy.T))
    w = np.maximum(w, 0.0)
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    return (V @ (np.sqrt(w)[:, None] * circ) * n_sigma).T + np.asarray(mean_xy)


# ==============================================================================================================
# 2. RUN REPORT
# Per-episode CSV, metrics and figures. Ground truth enters only through the episode log produced by run_simulation.py.
# ==============================================================================================================

LOG_COLUMNS = ["t", "phase", "x", "y", "th", "xh", "yh", "thh", "Pxx", "Pxy", "Pyy", "Pxt", "Pyt", "Ptt", "v_cmd",
               "w_cmd", "v_odom", "w_odom", "xr", "yr", "thr", "lidar_fix", "lidar_accepted", "lidar_nis",
               "detected", "accepted", "nis", "zx", "zy", "zyaw", "z_range", "z_view", "sx", "sy", "syaw", "frame_lost",
               "mx", "my", "mth", "mxh", "myh", "mthh", "mPxx", "mPxy", "mPyy", "mPxt", "mPyt", "mPtt"]


def cov_stack(L, prefix: str = "") -> np.ndarray:
    """Covariances of the navigation EKF (prefix "") or of the docking EKF (prefix "m")."""
    k = [prefix + n for n in ("Pxx", "Pxy", "Pyy", "Pxt", "Pyt", "Ptt")]
    return np.array([[[a, b, d], [b, c, e], [d, e, f]] for a, b, c, d, e, f in zip(*(L[n] for n in k))])


def est_errors(L) -> np.ndarray:
    E = np.column_stack([L["x"] - L["xh"], L["y"] - L["yh"], wrap(L["th"] - L["thh"])])
    return E


def dock_nees(L) -> np.ndarray:
    """NEES of the docking EKF (pose relative to the marker) on the steps where it runs."""
    m = np.isfinite(L["mxh"])
    E = np.column_stack([L["mx"] - L["mxh"], L["my"] - L["myh"], wrap(L["mth"] - L["mthh"])])[m]
    P = cov_stack(L, "m")[m]
    return np.array([e @ np.linalg.solve(p, e) for e, p in zip(E, P)])


def episode_metrics(res) -> dict:
    L = res.log
    E = est_errors(L)
    P = cov_stack(L)
    X = np.column_stack([L["x"], L["y"], L["th"]])
    Xh = np.column_stack([L["xh"], L["yh"], L["thh"]])
    ne = nees(X, Xh, P)
    nd = dock_nees(L)
    nis_l = np.array([u.nis for u in L["ekf_updates"]])
    nis_c = np.array([u.nis for u in L["dock_updates"]])
    moving = np.isin(L["phase"], ["track", "servo"])
    # Tracking error of the TRUE pose against the reference, in the reference frame (cross-track and heading).
    have_ref = moving & np.isfinite(L["xr"])
    c, s = np.cos(L["thr"][have_ref]), np.sin(L["thr"][have_ref])
    dx, dy = L["x"][have_ref] - L["xr"][have_ref], L["y"][have_ref] - L["yr"][have_ref]
    cross = -s * dx + c * dy
    head = wrap(L["th"][have_ref] - L["thr"][have_ref])
    lim = Limits()
    sat = (np.abs(L["v_cmd"]) >= lim.v_max - 1e-6) | (np.abs(L["w_cmd"]) >= lim.omega_max - 1e-6)
    gate = chi2.ppf(0.99, 3)
    return {
        "room": res.layout.name, "seed": res.cfg.seed, "status": res.status, "success": res.success,
        "dock_hint_error_m": float(res.layout.dock_y - res.layout.dock_hint_y),
        "dock_err_long_m": float(res.final_error[0]), "dock_err_lat_m": float(res.final_error[1]),
        "dock_err_pos_m": float(np.hypot(*res.final_error[:2])), "dock_err_heading_deg": float(np.rad2deg(res.final_error[2])),
        "time_s": res.duration,
        "rms_cross_track_m": float(np.sqrt(np.mean(cross ** 2))) if cross.size else np.nan,
        "rms_heading_track_deg": float(np.rad2deg(np.sqrt(np.mean(head ** 2)))) if head.size else np.nan,
        "control_effort": float(np.sum(L["v_cmd"] ** 2 + (0.1 * L["w_cmd"]) ** 2) * 0.1),
        # Feedback effort: the LQR correction on top of the planned speeds, and how fast the turn command changes.
        "rms_feedback_v": float(np.sqrt(np.nanmean(L["du_v"] ** 2))) if np.isfinite(L["du_v"]).any() else np.nan,
        "rms_feedback_w": float(np.sqrt(np.nanmean(L["du_w"] ** 2))) if np.isfinite(L["du_w"]).any() else np.nan,
        "rms_turn_rate_change": float(np.sqrt(np.mean((np.diff(L["w_cmd"][moving]) / 0.1) ** 2)))
        if moving.sum() > 1 else np.nan,
        "sat_fraction": float(sat[moving].mean()) if moving.any() else 0.0,
        "rms_est_pos_m": float(np.sqrt(np.mean(E[:, 0] ** 2 + E[:, 1] ** 2))),
        "max_est_pos_m": float(np.max(np.hypot(E[:, 0], E[:, 1]))),
        "rms_est_heading_deg": float(np.rad2deg(np.sqrt(np.mean(E[:, 2] ** 2)))),
        "mean_nees": float(ne.mean()), "frac_nees_in_95": float(np.mean((ne > chi2.ppf(0.025, 3)) & (ne < chi2.ppf(0.975, 3)))),
        "n_lidar_fixes": int(L["lidar_fix"].sum()), "n_lidar_accepted": int(L["lidar_accepted"].sum()),
        "mean_lidar_nis": float(nis_l.mean()) if nis_l.size else np.nan,
        "frac_lidar_gated": float(np.mean(nis_l >= gate)) if nis_l.size else np.nan,
        "dock_mean_nees": float(nd.mean()) if nd.size else np.nan,
        "n_detections": int(L["detected"].sum()), "n_accepted": int(L["accepted"].sum()),
        "mean_camera_nis": float(nis_c.mean()) if nis_c.size else np.nan,
        "frac_camera_gated": float(np.mean(nis_c >= gate)) if nis_c.size else np.nan,
        "final_dock_est_err_mm": float(1000 * np.hypot(*(res.final_error[:2] - res.final_est_error[:2])))
        if np.all(np.isfinite(res.final_est_error)) else np.nan,
        "n_replans": sum("replan" in e[1] for e in res.events),
        "n_lidar_replans": sum("LiDAR" in e[1] for e in res.events),
    }


def write_csv(res, path: Path) -> None:
    L = res.log
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        for k in range(len(L["t"])):
            w.writerow([L[c][k] if c == "phase" else f"{L[c][k]:.6g}" for c in LOG_COLUMNS])


def _camera_span(ax, L):
    """Shade the end of the run, where the camera is on (look and servo)."""
    cam = np.isin(L["phase"], ["look", "servo", "backup"])
    if cam.any():
        t = L["t"]
        ax.axvspan(t[cam][0], t[cam][-1], color="g", alpha=0.08, lw=0)


def plot_episode(res, out_dir: Path, title: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    L = res.log
    t = L["t"]
    E = est_errors(L)
    P = cov_stack(L)

    # 1. Paths on the map.
    fig, ax = plt.subplots(figsize=(8, 6.2))
    draw_map(ax, layout=res.layout)
    for _, plan in res.plans:
        ax.plot(plan.ref.x, plan.ref.y, color="0.6", lw=1, ls="--")
    ax.plot(L["x"], L["y"], "b", lw=1.6, label="true path")
    ax.plot(L["xh"], L["yh"], "r", lw=1.0, label="navigation EKF (odometry + LiDAR)")
    ax.plot(L["xd"], L["yd"], "m", lw=1.0, label="docking EKF (odometry + camera), drawn at the true marker")
    for k in range(0, len(t), 40):
        el = ellipse_points([L["xh"][k], L["yh"][k]], P[k][:2, :2])
        ax.plot(el[:, 0], el[:, 1], "r", lw=0.6, alpha=0.7)
    robot_triangle(ax, L["start_true"], "C0")
    robot_triangle(ax, L["final_true"], "b")
    ax.plot([], [], color="0.6", ls="--", label="planned paths")
    ax.plot([], [], "r", lw=0.6, label="3 sigma position ellipse (every 4 s)")
    ax.plot([], [], color="tab:purple", lw=6, alpha=0.25, label="given station position +- 0.20 m")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title or f"{res.layout.name}: {res.status}")
    fig.tight_layout()
    fig.savefig(out_dir / "path.png", dpi=130)
    plt.close(fig)

    # 2. Estimation errors of the navigation EKF with 3 sigma bounds.
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    sd = np.sqrt(np.stack([L["Pxx"], L["Pyy"], L["Ptt"]], 1))
    labels = ["x error [m]", "y error [m]", "heading error [deg]"]
    for i in range(3):
        sc = np.rad2deg(1) if i == 2 else 1.0
        ax[i].plot(t, E[:, i] * sc, "k", lw=1, label="true minus estimate")
        ax[i].fill_between(t, -3 * sd[:, i] * sc, 3 * sd[:, i] * sc, color="r", alpha=0.15, label="+-3 sigma")
        ax[i].set_ylabel(labels[i])
        _camera_span(ax[i], L)
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("navigation EKF (odometry + LiDAR position fix)")
    ax[2].set_xlabel("time [s] (green: camera on, visual servoing)")
    fig.tight_layout()
    fig.savefig(out_dir / "estimation_errors.png", dpi=130)
    plt.close(fig)

    # 3. NEES and NIS of both filters.
    X = np.column_stack([L["x"], L["y"], L["th"]])
    Xh = np.column_stack([L["xh"], L["yh"], L["thh"]])
    ne = nees(X, Xh, P)
    nd = dock_nees(L)
    td = t[np.isfinite(L["mxh"])]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].semilogy(t, np.maximum(ne, 1e-3), "r", lw=0.8, label=f"navigation EKF (mean {ne.mean():.2f})")
    if nd.size:
        ax[0].semilogy(td, np.maximum(nd, 1e-3), "m", lw=0.8, label=f"docking EKF (mean {nd.mean():.2f})")
    for q in (0.025, 0.975):
        ax[0].axhline(chi2.ppf(q, 3), color="k", ls="--", lw=0.8)
    ax[0].axhline(3, color="g", lw=0.8)
    ax[0].set_ylabel("NEES (3 dof)")
    ax[0].set_title("NEES (consistent: mean 3, dashed: 95 percent bounds)")
    ax[0].legend(fontsize=7)
    for upd, col, name in ((L["ekf_updates"], "r", "LiDAR fix"), (L["dock_updates"], "m", "camera")):
        if upd:
            tu = np.array([u.t for u in upd]); nu = np.array([u.nis for u in upd])
            ok = np.array([u.accepted for u in upd])
            ax[1].semilogy(tu[ok], nu[ok], ".", color=col, ms=3, label=f"{name} accepted")
            ax[1].semilogy(tu[~ok], nu[~ok], "x", color=col, ms=4, label=f"{name} gated out")
    ax[1].axhline(chi2.ppf(0.99, 3), color="k", ls="--", lw=0.8, label="gate (99 percent)")
    ax[1].legend(fontsize=7)
    ax[1].set_ylabel("NIS")
    ax[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "nees_nis.png", dpi=130)
    plt.close(fig)

    # 4. Control effort.
    lim = Limits()
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].plot(t, L["v_cmd"], "k", lw=1, label="command")
    ax[0].plot(t, L["v_odom"], "c", lw=0.7, label="odometry")
    ax[0].axhline(lim.v_max, color="r", ls="--", lw=0.8, label="limit")
    ax[0].set_ylabel("v [m/s]")
    ax[0].legend(fontsize=7)
    ax[1].plot(t, L["w_cmd"], "k", lw=1)
    ax[1].plot(t, L["w_odom"], "c", lw=0.7)
    for sgn in (1, -1):
        ax[1].axhline(sgn * lim.omega_max, color="r", ls="--", lw=0.8)
    ax[1].set_ylabel("omega [rad/s]")
    ax[1].set_xlabel("time [s]")
    phases = L["phase"]
    for a in ax:
        k0 = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or phases[k] != phases[k0]:
                a.text(t[k0], a.get_ylim()[1], phases[k0], fontsize=6, va="top")
                a.axvline(t[k0], color="0.8", lw=0.5)
                k0 = k
    fig.tight_layout()
    fig.savefig(out_dir / "control.png", dpi=130)
    plt.close(fig)

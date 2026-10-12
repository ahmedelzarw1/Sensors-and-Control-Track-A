"""Per-episode CSV, metrics and figures. Ground truth enters only through the episode log produced by runner.py."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from evaluation.metrics import nees  # noqa: E402
from evaluation.plotting import draw_map, ellipse_points, robot_triangle  # noqa: E402
from src.control.lqr import Limits  # noqa: E402
from src.tb3.interfaces import wrap  # noqa: E402

LOG_COLUMNS = ["t", "phase", "x", "y", "th", "xh", "yh", "thh", "Pxx", "Pxy", "Pyy", "Pxt", "Pyt", "Ptt", "v_cmd",
               "w_cmd", "v_odom", "w_odom", "xr", "yr", "thr", "detected", "accepted", "nis", "zx", "zy", "zyaw",
               "z_range", "z_view", "sx", "sy", "syaw", "frame_lost"]


def cov_stack(L) -> np.ndarray:
    return np.array([[[a, b, d], [b, c, e], [d, e, f]] for a, b, c, d, e, f in
                     zip(L["Pxx"], L["Pxy"], L["Pyy"], L["Pxt"], L["Pyt"], L["Ptt"])])


def est_errors(L) -> np.ndarray:
    E = np.column_stack([L["x"] - L["xh"], L["y"] - L["yh"], wrap(L["th"] - L["thh"])])
    return E


def hidden_drift(L) -> dict:
    """Estimation error against distance travelled since the last accepted marker fix (true path length).
    Only stretches that start from a fix are used, so the starting error is the post-fix error."""
    xy = np.column_stack([L["x"], L["y"]])
    step = np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)]
    E = est_errors(L)
    pos_err = np.hypot(E[:, 0], E[:, 1])
    sd = np.sqrt(np.maximum(L["Pxx"] + L["Pyy"], 0.0))
    dist, err, sig, head, seg = [], [], [], [], []
    acc = L["accepted"] == 1
    d, seen, sid = 0.0, False, -1
    for k in range(len(step)):
        if acc[k]:
            d, seen = 0.0, True
            sid += 1
            continue
        if not seen:
            continue
        d += step[k]
        dist.append(d); err.append(pos_err[k]); sig.append(sd[k]); head.append(abs(E[k, 2])); seg.append(sid)
    return {"dist": np.array(dist), "pos_err": np.array(err), "pos_sd": np.array(sig), "head_err": np.array(head),
            "segment": np.array(seg)}


def episode_metrics(res) -> dict:
    L = res.log
    E = est_errors(L)
    P = cov_stack(L)
    X = np.column_stack([L["x"], L["y"], L["th"]])
    Xh = np.column_stack([L["xh"], L["yh"], L["thh"]])
    ne = nees(X, Xh, P)
    nis = np.array([u.nis for u in L["ekf_updates"]]) if L["ekf_updates"] else np.array([])
    moving = np.isin(L["phase"], ["track", "dock"])
    # Tracking error of the TRUE pose against the reference, in the reference frame (cross-track and heading).
    have_ref = moving & np.isfinite(L["xr"])
    c, s = np.cos(L["thr"][have_ref]), np.sin(L["thr"][have_ref])
    dx, dy = L["x"][have_ref] - L["xr"][have_ref], L["y"][have_ref] - L["yr"][have_ref]
    cross = -s * dx + c * dy
    head = wrap(L["th"][have_ref] - L["thr"][have_ref])
    lim = Limits()
    sat = (np.abs(L["v_cmd"]) >= lim.v_max - 1e-6) | (np.abs(L["w_cmd"]) >= lim.omega_max - 1e-6)
    hd = hidden_drift(L)
    return {
        "start": res.cfg.start, "seed": res.cfg.seed, "status": res.status, "success": res.success,
        "dock_err_long_m": float(res.final_error[0]), "dock_err_lat_m": float(res.final_error[1]),
        "dock_err_pos_m": float(np.hypot(*res.final_error[:2])), "dock_err_heading_deg": float(np.rad2deg(res.final_error[2])),
        "time_s": res.duration,
        "rms_cross_track_m": float(np.sqrt(np.mean(cross ** 2))) if cross.size else np.nan,
        "rms_heading_track_deg": float(np.rad2deg(np.sqrt(np.mean(head ** 2)))) if head.size else np.nan,
        "control_effort": float(np.sum(L["v_cmd"] ** 2 + (0.1 * L["w_cmd"]) ** 2) * 0.1),
        "sat_fraction": float(sat[moving].mean()) if moving.any() else 0.0,
        "rms_est_pos_m": float(np.sqrt(np.mean(E[:, 0] ** 2 + E[:, 1] ** 2))),
        "max_est_pos_m": float(np.max(np.hypot(E[:, 0], E[:, 1]))),
        "rms_est_heading_deg": float(np.rad2deg(np.sqrt(np.mean(E[:, 2] ** 2)))),
        "mean_nees": float(ne.mean()), "frac_nees_in_95": float(np.mean((ne > chi2.ppf(0.025, 3)) & (ne < chi2.ppf(0.975, 3)))),
        "n_detections": int(L["detected"].sum()), "n_accepted": int(L["accepted"].sum()),
        "mean_nis": float(nis.mean()) if nis.size else np.nan,
        "frac_gated": float(np.mean(nis >= chi2.ppf(0.99, 3))) if nis.size else np.nan,
        "max_hidden_dist_m": float(hd["dist"].max()) if hd["dist"].size else 0.0,
        "max_hidden_pos_err_m": float(hd["pos_err"].max()) if hd["pos_err"].size else 0.0,
        "n_replans": sum("replan" in e[1] for e in res.events),
    }


def write_csv(res, path: Path) -> None:
    L = res.log
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_COLUMNS)
        for k in range(len(L["t"])):
            w.writerow([L[c][k] if c == "phase" else f"{L[c][k]:.6g}" for c in LOG_COLUMNS])


def _fix_spans(ax, L):
    acc = L["accepted"] == 1
    t = L["t"]
    for k in np.flatnonzero(acc):
        ax.axvspan(t[k] - 0.05, t[k] + 0.05, color="g", alpha=0.08, lw=0)


def plot_episode(res, out_dir: Path, title: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    L = res.log
    t = L["t"]
    E = est_errors(L)
    P = cov_stack(L)

    # 1. Paths on the map.
    fig, ax = plt.subplots(figsize=(8, 6.2))
    draw_map(ax, layout=res.cfg.layout)
    for _, plan in res.plans:
        ax.plot(plan.ref.x, plan.ref.y, color="0.6", lw=1, ls="--")
    ax.plot(L["x"], L["y"], "b", lw=1.6, label="true path")
    ax.plot(L["xh"], L["yh"], "r", lw=1.0, label="estimated path (EKF)")
    acc = L["accepted"] == 1
    ax.plot(L["xh"][acc], L["yh"][acc], "g.", ms=2.5, label="marker fix accepted")
    for k in range(0, len(t), 40):
        el = ellipse_points([L["xh"][k], L["yh"][k]], P[k][:2, :2])
        ax.plot(el[:, 0], el[:, 1], "r", lw=0.6, alpha=0.7)
    robot_triangle(ax, L["start_true"], "C0")
    robot_triangle(ax, L["final_true"], "b")
    ax.plot([], [], color="0.6", ls="--", label="planned references")
    ax.plot([], [], "r", lw=0.6, label="3 sigma position ellipse (every 4 s)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title or f"{res.cfg.start}: {res.status}")
    fig.tight_layout()
    fig.savefig(out_dir / "path.png", dpi=130)
    plt.close(fig)

    # 2. Estimation errors with 3 sigma bounds.
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    sd = np.sqrt(np.stack([L["Pxx"], L["Pyy"], L["Ptt"]], 1))
    labels = ["x error [m]", "y error [m]", "heading error [deg]"]
    for i in range(3):
        sc = np.rad2deg(1) if i == 2 else 1.0
        ax[i].plot(t, E[:, i] * sc, "k", lw=1, label="true minus estimate")
        ax[i].fill_between(t, -3 * sd[:, i] * sc, 3 * sd[:, i] * sc, color="r", alpha=0.15, label="+-3 sigma")
        ax[i].set_ylabel(labels[i])
        _fix_spans(ax[i], L)
    ax[0].legend(fontsize=7, loc="upper right")
    ax[2].set_xlabel("time [s] (green: marker fix accepted)")
    fig.tight_layout()
    fig.savefig(out_dir / "estimation_errors.png", dpi=130)
    plt.close(fig)

    # 3. NEES and NIS.
    X = np.column_stack([L["x"], L["y"], L["th"]])
    Xh = np.column_stack([L["xh"], L["yh"], L["thh"]])
    ne = nees(X, Xh, P)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax[0].semilogy(t, np.maximum(ne, 1e-3), "k", lw=0.8)
    for q in (0.025, 0.975):
        ax[0].axhline(chi2.ppf(q, 3), color="r", ls="--", lw=0.8)
    ax[0].axhline(3, color="g", lw=0.8)
    ax[0].set_ylabel("NEES (3 dof)")
    ax[0].set_title(f"mean NEES {ne.mean():.2f} (consistent: 3, dashed: 95 percent bounds)")
    upd = L["ekf_updates"]
    if upd:
        tu = np.array([u.t for u in upd]); nu = np.array([u.nis for u in upd]); ok = np.array([u.accepted for u in upd])
        ax[1].semilogy(tu[ok], nu[ok], "g.", ms=3, label="accepted")
        ax[1].semilogy(tu[~ok], nu[~ok], "rx", ms=4, label="gated out")
        ax[1].axhline(chi2.ppf(0.99, 3), color="r", ls="--", lw=0.8, label="gate (99 percent)")
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


def plot_drift(results, path: Path) -> dict:
    """Position and heading error against distance travelled since the last marker fix, pooled over the episodes that
    docked. A failed run (for example a robot stuck against an obstacle) would swamp the statistics, and its cause is
    reported separately."""
    D, Ep, S, H = [], [], [], []
    for r in [r for r in results if r.success]:
        hd = hidden_drift(r.log)
        D.append(hd["dist"]); Ep.append(hd["pos_err"]); S.append(hd["pos_sd"]); H.append(hd["head_err"])
    D, Ep, S, H = map(np.concatenate, (D, Ep, S, H))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].scatter(D, Ep * 1e3, s=2, alpha=0.4, label="position error")
    ax[0].scatter(D, 3 * S * 1e3, s=2, alpha=0.4, c="r", label="3 sigma (EKF)")
    ax[0].set(xlabel="distance travelled since the last marker fix [m]", ylabel="[mm]",
              title="growth of the estimation error while the marker is hidden")
    ax[0].legend(fontsize=8)
    ax[1].scatter(D, np.rad2deg(H), s=2, alpha=0.4)
    ax[1].set(xlabel="distance since last fix [m]", ylabel="heading error [deg]")
    out = {}
    edges = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (D >= lo) & (D < hi)
        if sel.sum() >= 5:
            rows.append({"dist_m": f"{lo:.2f}-{hi:.2f}", "n": int(sel.sum()),
                         "median_pos_err_mm": float(np.median(Ep[sel]) * 1e3),
                         "p95_pos_err_mm": float(np.percentile(Ep[sel], 95) * 1e3),
                         "median_3sigma_mm": float(np.median(3 * S[sel]) * 1e3),
                         "frac_within_3sigma": float(np.mean(Ep[sel] <= 3 * S[sel])),
                         "median_heading_err_deg": float(np.rad2deg(np.median(H[sel])))})
    out["bins"] = rows
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out

"""Trade-off studies: what each design choice buys and what it costs, measured on the same 12 random rooms
(seeds 400 to 411, not used for any calibration or tuning).

1. LQR input weight (control effort against tracking error). R is multiplied by 0.25, 1 (the design), 4 and 16.
   Measured: feedback effort (RMS of the LQR turn-rate correction on top of the planned turn rate, and RMS rate of
   change of the turn command), RMS cross-track error of the TRUE pose against the reference, docking error.
2. LiDAR fix covariance (estimator consistency against responsiveness). lidar_r_scale 0.5, 1.25 (the tuned value), 2.5,
   5 and 10, with a push at t = 8 s (+0.15 m/s, +1.0 rad/s) so the filter has a sudden error to correct.
   Measured: mean NEES (3 = consistent), RMS estimation error, and the time for the estimation error caused by the push
   to fall to half its peak (how quickly the filter believes the LiDAR over the odometry).
3. LiDAR position fix on or off (the value of the sensor fusion). Off: the navigation EKF runs on odometry alone for
   the whole trip (the LiDAR map and safety stop still run, placed with the drifting estimate).
   Measured: estimation error at the end of the trip, success, collisions.
4. LQR gain: one fixed gain, the gain scheduled on the reference (the design), or finite-horizon time-varying LQR.

Run: python -m experiments.tradeoff_studies [workers] [study names to rerun only those]
     python -m experiments.tradeoff_studies --report      (tables and figure again from runs.json)
Output: results/tradeoffs/summary.csv, runs.json, tradeoffs.png
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.robustness_trials import summarise
from experiments.run_simulation import EpisodeConfig, run_episode
from robot_code import random_room as rl
from robot_code.kalman_filter import EKFParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tradeoffs"
SEEDS = tuple(range(400, 412))
PUSH_T = 8.0

STUDIES = {
    "lqr_r_scale": [0.25, 1.0, 4.0, 16.0],
    "lidar_r_scale": [0.5, 1.25, 2.5, 5.0, 10.0],
    "lidar_fix": [True, False],
    "tracker_mode": ["fixed", "scheduled", "tv"],
}


def push_recovery(L) -> tuple[float, float]:
    """(peak extra estimation error after the push in mm, time in s for it to fall to half). NaN if the push caused
    less than 5 mm of extra error."""
    t = L["t"]
    err = np.hypot(L["x"] - L["xh"], L["y"] - L["yh"])
    pre = err[(t >= PUSH_T - 2.0) & (t < PUSH_T)]
    base = float(np.median(pre)) if pre.size else 0.0
    win = np.flatnonzero((t >= PUSH_T) & (t <= PUSH_T + 3.0))
    if not win.size:
        return np.nan, np.nan
    kp = win[np.argmax(err[win])]
    peak = err[kp] - base
    if peak < 0.005:
        return 1000 * peak, np.nan
    after = np.flatnonzero((np.arange(len(t)) > kp) & (err - base <= 0.5 * peak))
    return 1000 * peak, float(t[after[0]] - PUSH_T) if after.size else np.nan


def one(job):
    study, value, seed = job
    layout, _ = rl.sample_layout(seed)
    kw = {}
    if study == "lqr_r_scale":
        kw["lqr_r_scale"] = value
    elif study == "lidar_r_scale":
        kw.update(ekf=EKFParams.calibrated(lidar_r_scale=value), impulses=((PUSH_T, 0.15, 1.0),))
    elif study == "lidar_fix":
        kw["lidar_fix"] = value
    else:
        kw["tracker_mode"] = value
    try:
        res = run_episode(EpisodeConfig(layout=layout, seed=seed, **kw))
    except Exception as e:  # noqa: BLE001 (reported, not hidden)
        return {"study": study, "value": value, "seed": seed, "status": f"error: {e!r}", "success": False}
    out = summarise(res)
    out.update(study=study, value=value)
    if study == "lidar_r_scale":
        out["push_peak_mm"], out["push_half_recovery_s"] = push_recovery(res.log)
    return out


def _agg(rows):
    med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in rows]))
    mean = lambda k: float(np.nanmean([r.get(k, np.nan) for r in rows]))
    ok = [r for r in rows if r["success"]]
    return {"runs": len(rows), "docked": len(ok),
            "median_dock_err_mm": round(float(np.median([1000 * r["dock_err_pos_m"] for r in ok])), 2) if ok else None,
            "mean_rms_feedback_v_mm_s": round(1000 * mean("rms_feedback_v"), 2),
            "mean_rms_feedback_w": round(mean("rms_feedback_w"), 4),
            "mean_rms_turn_rate_change": round(mean("rms_turn_rate_change"), 3),
            "mean_rms_cross_track_mm": round(1000 * mean("rms_cross_track_m"), 2),
            "mean_sat_fraction": round(mean("sat_fraction"), 4),
            "mean_nees": round(mean("mean_nees"), 3),
            "mean_rms_est_err_mm": round(1000 * mean("rms_est_pos_m"), 2),
            "median_est_err_end_of_trip_mm": round(med("est_err_end_of_trip_mm"), 1),
            "median_push_peak_mm": round(med("push_peak_mm"), 1),
            "median_push_half_recovery_s": round(med("push_half_recovery_s"), 2),
            "median_time_s": round(med("time_s"), 1),
            "runs_with_collision": sum(r.get("contact_steps", 0) > 0 for r in rows),
            "failures": "; ".join(f"seed {r['seed']}: {r['status']}" for r in rows if not r["success"])}


def plot(summary, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    S = lambda st: [s for s in summary if s["study"] == st]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    def scatter(a, xs, ys, labels, design, ordered=True, offsets=None):
        if ordered:   # the swept values are ordered, so the line shows the trend
            a.plot(xs, ys, color=blue, lw=1, alpha=0.5)
        for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
            a.plot(x, y, "o", color=blue, ms=9 if lab == design else 7, mec=ink if lab == design else "white", mew=1.5)
            a.annotate(lab + (" (design)" if lab == design else ""), (x, y), textcoords="offset points",
                       xytext=(offsets or {}).get(i, (7, 5)), fontsize=8, color=ink)

    s = S("lqr_r_scale")
    scatter(ax[0, 0], [r["mean_rms_feedback_w"] for r in s], [r["mean_rms_cross_track_mm"] for r in s],
            [f"R x {r['value']:g}" for r in s], "R x 1")
    ax[0, 0].set(xlabel="RMS LQR turn-rate correction [rad/s]", ylabel="RMS cross-track error [mm]",
                 title="1. LQR input weight: control effort against tracking error")
    s = S("lidar_r_scale")
    scatter(ax[0, 1], [r["mean_nees"] for r in s], [r["median_push_half_recovery_s"] for r in s],
            [f"R_lidar x {r['value']:g}" for r in s], "R_lidar x 1.25", offsets={0: (-20, 12), 1: (-30, 12), 2: (-30, 12)})
    ax[0, 1].axvline(3.0, color=muted, ls=":", lw=1)
    ax[0, 1].text(3.0, ax[0, 1].get_ylim()[1], " consistent (NEES 3)", fontsize=7, color=muted, va="top")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set(xlabel="mean NEES of the navigation EKF (log scale)",
                 ylabel="time to halve the estimation error after a push [s]",
                 title="2. LiDAR fix covariance: consistency against responsiveness")
    s = S("lidar_fix")
    labels = ["LiDAR fix on (design)" if r["value"] else "odometry only" for r in s]
    vals = [r["median_est_err_end_of_trip_mm"] for r in s]
    ax[1, 0].barh(labels, vals, color=blue, height=0.45)
    for i, (r, v) in enumerate(zip(s, vals)):
        ax[1, 0].text(1.1, i - 0.3, f"{v:.0f} mm; docked {r['docked']}/{r['runs']}; runs with a collision: "
                      f"{r['runs_with_collision']}", va="center", fontsize=8, color=ink)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlim(1, 1000)
    ax[1, 0].set_ylim(-0.75, 1.4)
    ax[1, 0].set(xlabel="estimation error at the end of the trip, median [mm] (log scale)",
                 title="3. Value of the LiDAR position fix")
    s = S("tracker_mode")
    scatter(ax[1, 1], [r["mean_rms_feedback_w"] for r in s], [r["mean_rms_cross_track_mm"] for r in s],
            [str(r["value"]) for r in s], "scheduled", ordered=False)
    ax[1, 1].set(xlabel="RMS LQR turn-rate correction [rad/s]", ylabel="RMS cross-track error [mm]",
                 title="4. LQR gain: fixed, scheduled, time-varying")
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color=grid, lw=0.8)
        a.set_axisbelow(True)
        a.title.set_fontsize(10)
    fig.suptitle(f"Trade-offs measured on {len(SEEDS)} random rooms (seeds {SEEDS[0]} to {SEEDS[-1]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def report(rows) -> None:
    summary = [{"study": st, "value": v, **_agg([r for r in rows if r["study"] == st and r["value"] == v])}
               for st, vals in STUDIES.items() for v in vals]
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    plot(summary, OUT / "tradeoffs.png")
    for s in summary:
        print(s)


def main(workers: int = 6, only: tuple = ()) -> None:
    """only: rerun just these studies and keep the saved runs of the others."""
    OUT.mkdir(parents=True, exist_ok=True)
    studies = {k: v for k, v in STUDIES.items() if not only or k in only}
    jobs = [(st, v, s) for st, vals in studies.items() for v in vals for s in SEEDS]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(one, jobs))
    if only and (OUT / "runs.json").exists():
        rows = [r for r in json.loads((OUT / "runs.json").read_text()) if r["study"] not in only] + rows
    (OUT / "runs.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    report(rows)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(json.loads((OUT / "runs.json").read_text()))
        sys.exit()
    args = [a for a in sys.argv[1:] if not a.isdigit()]
    nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(workers=nums[0] if nums else 6, only=tuple(args))

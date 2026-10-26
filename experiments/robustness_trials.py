"""Robustness trials: the full mission in random rooms under disturbances and extra noise.

Every condition runs on the same 20 random rooms (seeds 300 to 319, not used for any calibration or tuning), so the
conditions are compared on identical rooms, start points and stations. Each condition changes the simulated world or
the sensor stream only; the robot's software and its settings are unchanged, so it does not know about the change.

Conditions (all values are our choices, picked to be clearly beyond the nominal settings):
- odometry noise x2 and x4         encoder velocity noise std doubled / quadrupled
- wheel radius +-2 percent         left wheel 2 percent larger, right 2 percent smaller than the software believes
- floor friction x0.3              wheel and floor sliding friction cut to 30 percent (wheel slip)
- push while navigating            velocity kick at t = 8 s: +0.15 m/s forward and +1.0 rad/s turn
- push while docking               the same kind of kick (+0.10 m/s, +0.8 rad/s) 3 s into the visual servoing
- camera frames lost 50 percent    every camera frame is dropped with probability 0.5
- LiDAR noise x2                   range noise twice the datasheet values the robot assumes
- start 3x further off             start placement error three times the prior standard deviation the robot assumes
- combined                         odometry x2, wheels +-1 percent, friction x0.5, LiDAR x1.5, 30 percent frames lost
- station 0.35 / 0.5 / 0.7 m off   limit tests: the true station 0.35, 0.5 or 0.7 m from the given position, beyond the
                                   +-0.20 m the robot assumes. From the end point (0.9 m from the wall) the camera sees
                                   about +-0.69 m along the wall, so 0.7 m starts outside the first view.

A trial is INVALID (not counted) when the simulator cannot place the robot: MuJoCo reports contact at the start (this
happens only with the 3x start error, when the placement lands inside a wall).

Reported per condition: success rate with its 95 percent Wilson interval, docking error, time, the number of runs with
any collision (robot touching an obstacle or a wall, from MuJoCo contacts), and for every failure the last events of
the mission, so the cause can be traced.

Run: python -m experiments.robustness_trials [workers] [condition names to run only those]
     python -m experiments.robustness_trials --report      (tables and figure again from runs.json)
Output: results/robustness/summary.csv, runs.json, robustness.png
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.run_simulation import EpisodeConfig, run_episode
from experiments.scores import wilson_interval
from robot_code import random_room as rl
from robot_code.simulator import OdomNoise

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "robustness"
SEEDS = tuple(range(300, 320))

CONDITIONS = {
    "nominal": {},
    "odometry noise x2": dict(odom_noise=OdomNoise(0.010, 0.020)),
    "odometry noise x4": dict(odom_noise=OdomNoise(0.020, 0.040)),
    "wheel radius +-2 percent": dict(wheel_scale=(1.02, 0.98)),
    "floor friction x0.3": dict(friction_scale=0.3),
    "push while navigating": dict(impulses=((8.0, 0.15, 1.0),)),
    "push while docking": dict(servo_push=(3.0, 0.10, 0.8)),
    "camera frames lost 50 percent": dict(marker_dropout=0.5),
    "LiDAR noise x2": dict(lidar_noise_scale=2.0),
    "start 3x further off": dict(start_offset_scale=3.0),
    "combined": dict(odom_noise=OdomNoise(0.010, 0.020), wheel_scale=(1.01, 0.99), friction_scale=0.5,
                     lidar_noise_scale=1.5, marker_dropout=0.3),
    "station 0.35 m off (limit test)": dict(hint_offset=0.35),
    "station 0.5 m off (limit test)": dict(hint_offset=0.5),
    "station 0.7 m off (limit test)": dict(hint_offset=0.7),
}


def invalid(r) -> bool:
    return str(r.get("status", "")).startswith("error: RuntimeError(\"robot placed in contact")


def summarise(res) -> dict:
    """The numbers kept from one episode (shared with experiments/tradeoff_studies.py)."""
    from experiments.plots import episode_metrics
    L = res.log
    m = episode_metrics(res)
    err = np.hypot(L["x"] - L["xh"], L["y"] - L["yh"])
    look = np.flatnonzero(L["phase"] == "look")
    m.update(contact_steps=int(L["contact_steps"]), contacts=L["contact_names"],
             est_err_end_of_trip_mm=float(1000 * err[look[0]]) if look.size else np.nan,
             n_reinit=len(L["reinit_events"]), last_events=[e for e in res.events][-4:])
    return m


def one(job):
    name, seed = job
    kw = dict(CONDITIONS[name])
    layout, _ = rl.sample_layout(seed, hint_offset=kw.pop("hint_offset", None))
    try:
        res = run_episode(EpisodeConfig(layout=layout, seed=seed, **kw))
    except Exception as e:  # noqa: BLE001 (reported, not hidden)
        return {"condition": name, "seed": seed, "status": f"error: {e!r}", "success": False}
    out = summarise(res)
    out["condition"] = name
    return out


def plot(rows, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(CONDITIONS)
    blue, ink, muted = "#2a78d6", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        rr = [r for r in rows if r["condition"] == n and not invalid(r)]
        n_bad = sum(invalid(r) for r in rows if r["condition"] == n)
        k = sum(r["success"] for r in rr)
        lo, hi = wilson_interval(k, len(rr))
        ax[0].plot([100 * lo, 100 * hi], [yi, yi], color=blue, lw=2, solid_capstyle="round")
        ax[0].plot(100 * k / len(rr), yi, "o", color=blue, ms=8, mec="white", mew=2)
        ax[0].text(106, yi, f"{k}/{len(rr)}" + (f" ({n_bad} invalid)" if n_bad else ""), va="center", fontsize=8,
                   color=ink)
        pos = np.array([1000 * r["dock_err_pos_m"] for r in rr if r.get("status") in ("success", "out_of_tolerance")])
        if pos.size:
            jitter = (np.random.default_rng(0).random(pos.size) - 0.5) * 0.35
            ax[1].plot(pos, yi + jitter, "o", color=blue, ms=4, alpha=0.6, mec="none")
            ax[1].plot(np.median(pos), yi, "|", color=ink, ms=16, mew=2)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names, fontsize=9, color=ink)
    ax[0].set_xlim(0, 135)
    ax[0].set_xticks([0, 20, 40, 60, 80, 100])
    ax[0].set_xlabel("docked within tolerance [percent], 95 percent Wilson interval", color=muted)
    ax[1].axvline(np.hypot(20, 30), color=muted, ls=":", lw=1)
    ax[1].text(np.hypot(20, 30), y[0] + 0.6, "tolerance corner (30 mm, 20 mm) ", fontsize=7, color=muted, ha="right")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("final docking position error [mm] (log scale; bar: median)", color=muted)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="x", color="#e4e3df", lw=0.8)
        a.set_axisbelow(True)
    fig.suptitle(f"Robustness: {len(SEEDS)} random rooms per condition (seeds {SEEDS[0]} to {SEEDS[-1]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(workers: int = 6, only: tuple = ()) -> None:
    """only: run just these conditions and keep the saved runs of the others."""
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s) for n in CONDITIONS for s in SEEDS if not only or n in only]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(one, jobs))
    if only and (OUT / "runs.json").exists():
        rows = [r for r in json.loads((OUT / "runs.json").read_text()) if r["condition"] not in only] + rows
    rows.sort(key=lambda r: (list(CONDITIONS).index(r["condition"]), r["seed"]))
    (OUT / "runs.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    report(rows)


def report(rows) -> None:
    summary = []
    for n in CONDITIONS:
        n_bad = sum(invalid(r) for r in rows if r["condition"] == n)
        rr = [r for r in rows if r["condition"] == n and not invalid(r)]
        k = sum(r["success"] for r in rr)
        lo, hi = wilson_interval(k, len(rr))
        pos = [1000 * r["dock_err_pos_m"] for r in rr if r["success"]]
        summary.append({"condition": n, "runs": len(rr), "invalid_placements": n_bad, "docked": k, "wilson_lo_pct": round(100 * lo, 1),
                        "wilson_hi_pct": round(100 * hi, 1),
                        "median_dock_err_mm": round(float(np.median(pos)), 1) if pos else None,
                        "max_dock_err_mm": round(float(np.max(pos)), 1) if pos else None,
                        "max_heading_err_deg": round(max(abs(r["dock_err_heading_deg"]) for r in rr if r["success"]), 2)
                        if pos else None,
                        "median_time_s": round(float(np.median([r["time_s"] for r in rr if "time_s" in r])), 1),
                        "runs_with_collision": sum(r.get("contact_steps", 0) > 0 for r in rr),
                        "median_est_err_end_of_trip_mm": round(float(np.nanmedian(
                            [r.get("est_err_end_of_trip_mm", np.nan) for r in rr])), 1),
                        "mean_nees": round(float(np.mean([r["mean_nees"] for r in rr])), 2),
                        "mean_dock_nees": round(float(np.nanmean([r["dock_mean_nees"] for r in rr])), 2),
                        "failures": "; ".join(f"seed {r['seed']}: {r['status']}" for r in rr if not r["success"])})
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    plot(rows, OUT / "robustness.png")
    for s in summary:
        print(s)
    print("\nFailures and collisions:")
    for r in rows:
        if not r["success"] or r.get("contact_steps", 0):
            print(r["condition"], r["seed"], r["status"], "contacts", r.get("contact_steps"), r.get("contacts"),
                  r.get("last_events"))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(json.loads((OUT / "runs.json").read_text()))
    else:
        nums = [int(a) for a in sys.argv[1:] if a.isdigit()]
        main(workers=nums[0] if nums else 6, only=tuple(a for a in sys.argv[1:] if not a.isdigit()))

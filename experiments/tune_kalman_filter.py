"""Consistency tuning of the two EKFs (NEES and NIS based).

The measurement covariances come from models: the LiDAR fix from least squares with the LDS-01 datasheet noise
(robot_code/lidar.py), the camera from the measured noise model (robot_code/vision.py). Effects the models leave out
(errors of consecutive scans or frames that are correlated, the body rocking) make a filter overconfident or
overcautious. This script runs the full mission in random rooms for several scale factors and measures the consistency
of both filters against ground truth:

- Navigation EKF (room frame, odometry + LiDAR fix): NEES of the pose estimate over the whole run.
- Docking EKF (marker frame, odometry + camera): NEES of the pose relative to the marker while it runs.
A consistent filter with a 3-dimensional state has mean NEES 3. The NIS of the updates and the fraction rejected by the
99 percent gate are reported as well.

Swept: lidar_r_scale (multiplies the std of the LiDAR fix) and camera_r_scale (multiplies the std of the camera
measurement). The extra odometry process noise q_inflation is kept at Q_INFLATION, the value chosen by the earlier
consistency tuning of the odometry model. Rule: among the settings with every run docked and at most MAX_REJECTED of
the LiDAR fixes and of the camera measurements rejected by the gate (the 99 percent gate alone rejects 1 percent), the
pair whose two mean NEES values are closest to 3. Ground truth is used for NEES here only.

Run: python -m experiments.tune_kalman_filter            (all runs, then the choice)
     python -m experiments.tune_kalman_filter --choose   (the choice again from results/calibration/kalman_filter_tuning.json)
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from experiments.scores import nees
from experiments.run_simulation import EpisodeConfig, run_episode
from robot_code import random_room as rl
from robot_code.data_types import wrap
from robot_code.kalman_filter import EKFParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "calibration"
TUNING_JSON = ROOT / "settings" / "kalman_filter_tuning.json"
Q_INFLATION = 5.0
GATE = 11.345            # chi-square 99 percent, 3 dof
MAX_REJECTED = 0.02      # largest accepted fraction of measurements rejected by the gate


def _cov(L, prefix=""):
    k = [prefix + n for n in ("Pxx", "Pxy", "Pyy", "Pxt", "Pyt", "Ptt")]
    return np.array([[[a, b, d], [b, c, e], [d, e, f]] for a, b, c, d, e, f in zip(*(L[n] for n in k))])


def one(args):
    lidar_scale, cam_scale, seed = args
    layout, _ = rl.sample_layout(seed)
    r = run_episode(EpisodeConfig(layout=layout, seed=seed,
                                  ekf=EKFParams.calibrated(q_inflation=Q_INFLATION, lidar_r_scale=lidar_scale,
                                                           use_tuning=False),
                                  vision_r_scale=(cam_scale,) * 3, use_tuning=False))
    L = r.log
    nav = nees(np.column_stack([L["x"], L["y"], L["th"]]), np.column_stack([L["xh"], L["yh"], L["thh"]]), _cov(L))
    m = np.isfinite(L["mxh"])
    E = np.column_stack([L["mx"] - L["mxh"], L["my"] - L["myh"], wrap(L["mth"] - L["mthh"])])[m]
    P = _cov(L, "m")[m]
    dock = np.array([e @ np.linalg.solve(p, e) for e, p in zip(E, P)])
    nis_l = np.array([u.nis for u in L["ekf_updates"]])
    nis_c = np.array([u.nis for u in L["dock_updates"]])
    return {"lidar_r_scale": lidar_scale, "camera_r_scale": cam_scale, "seed": seed, "status": r.status,
            "nav_nees": float(nav.mean()), "dock_nees": float(dock.mean()) if dock.size else None,
            "lidar_nis": float(nis_l.mean()), "lidar_rejected": float(np.mean(nis_l >= GATE)),
            "camera_nis": float(nis_c.mean()) if nis_c.size else None,
            "camera_rejected": float(np.mean(nis_c >= GATE)) if nis_c.size else None,
            "final_error": r.final_error.tolist(), "duration": r.duration}


GRID = [(lr, cr) for lr in (1.0, 1.25, 1.5, 2.0) for cr in (0.5, 0.75, 1.0)]


def main(grid=GRID, seeds=tuple(range(200, 208)), workers: int = 6) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(lr, cr, sd) for lr, cr in grid for sd in seeds]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(one, jobs))
    summary = []
    for lr, cr in grid:
        rr = [r for r in rows if r["lidar_r_scale"] == lr and r["camera_r_scale"] == cr]
        mean = lambda k: float(np.mean([r[k] for r in rr if r[k] is not None]))
        summary.append({"lidar_r_scale": lr, "camera_r_scale": cr, "n": len(rr),
                        "success": sum(r["status"] == "success" for r in rr),
                        "nav_nees": mean("nav_nees"), "dock_nees": mean("dock_nees"),
                        "lidar_nis": mean("lidar_nis"), "lidar_rejected": mean("lidar_rejected"),
                        "camera_nis": mean("camera_nis"), "camera_rejected": mean("camera_rejected")})
    (OUT / "kalman_filter_tuning.json").write_text(json.dumps({"summary": summary, "runs": rows, "seeds": list(seeds)},
                                                              indent=2) + "\n")
    choose(summary, seeds)


def choose(summary, seeds) -> None:
    for s in summary:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
    ok = [s for s in summary if s["success"] == s["n"] and s["lidar_rejected"] <= MAX_REJECTED
          and s["camera_rejected"] <= MAX_REJECTED]
    if not ok:
        print("no setting met the rule; settings/kalman_filter_tuning.json left unchanged")
        return
    best = min(ok, key=lambda s: abs(s["nav_nees"] - 3.0) + abs(s["dock_nees"] - 3.0))
    TUNING_JSON.write_text(json.dumps({
        "source": f"experiments/tune_kalman_filter.py, random rooms seeds {seeds[0]} to {seeds[-1]} per setting",
        "rule": f"among the settings with every run docked and at most {MAX_REJECTED:.0%} of the LiDAR fixes and camera "
                "measurements gated out, the (lidar_r_scale, camera_r_scale) whose mean NEES of the navigation EKF and "
                "of the docking EKF are closest to 3; q_inflation kept from the odometry tuning",
        "q_inflation": Q_INFLATION, "lidar_r_scale": best["lidar_r_scale"], "camera_r_scale": best["camera_r_scale"],
        "nav_nees": best["nav_nees"], "dock_nees": best["dock_nees"], "lidar_nis": best["lidar_nis"],
        "camera_nis": best["camera_nis"], "lidar_rejected": best["lidar_rejected"],
        "camera_rejected": best["camera_rejected"]}, indent=2) + "\n")
    print("chosen:", best)


if __name__ == "__main__":
    if "--choose" in sys.argv:
        saved = json.loads((OUT / "kalman_filter_tuning.json").read_text())
        choose(saved["summary"], saved.get("seeds", [r["seed"] for r in saved["runs"]]))
    else:
        main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 6)

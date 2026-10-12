"""Consistency tuning of the EKF extra process noise (NIS and NEES based).

The odometry calibration (evaluation/odometry_calibration.py) measures the extra process noise on short open-loop
manoeuvres. Closed-loop runs contain effects those manoeuvres underrepresent (turns in place during the marker search,
start-up from rest, correlated slip), so the measured values are multiplied by an inflation factor. This script runs
the docking mission for several inflation factors and reports the filter consistency:

- NIS of accepted and rejected marker updates. A consistent filter with 3-dimensional measurements has mean NIS 3, and
  about 1 percent above the 99 percent gate (11.34).
- NEES of the pose estimate against ground truth (evaluation only). Mean 3 when consistent.

The same sweep scales the yaw measurement standard deviation. The yaw errors measured in motion are correlated over
many frames (the body rocks slowly), which a filter that assumes independent errors treats as more information than it
is. Scaling the yaw R is the standard remedy. The pair chosen is the smallest that gives a mean NEES near 3 with a gate
rejection rate near 1 percent.
Ground truth is used for NEES here only.

Run: python -m evaluation.tune_ekf
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from evaluation.metrics import nees
from evaluation.runner import EpisodeConfig, run_episode
from src import known_map as km
from src.estimation.ekf import EKFParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "calibration"
TUNING_JSON = ROOT / "config" / "ekf_tuning.json"
NEES_TOL = 0.25          # |mean NEES - 3| accepted as consistent
MAX_REJECTED = 0.02      # gate rejection rate accepted (the 99 percent gate alone rejects 1 percent)


def one(args):
    infl, yaw_scale, start, seed = args
    r = run_episode(EpisodeConfig(start=start, seed=seed, ekf=EKFParams.calibrated(q_inflation=infl),
                                  vision_r_scale=(1.0, 1.0, yaw_scale), use_tuning=False))
    L = r.log
    X = np.column_stack([L["x"], L["y"], L["th"]])
    Xh = np.column_stack([L["xh"], L["yh"], L["thh"]])
    P = np.array([[[a, b, d], [b, c, e], [d, e, f]] for a, b, c, d, e, f in
                  zip(L["Pxx"], L["Pxy"], L["Pyy"], L["Pxt"], L["Pyt"], L["Ptt"])])
    ne = nees(X, Xh, P)
    nis = np.array([u.nis for u in L["ekf_updates"]])
    return {"inflation": infl, "yaw_r_scale": yaw_scale, "start": start, "seed": seed, "status": r.status,
            "mean_nees": float(ne.mean()), "frac_nees_gt_chi2_99": float(np.mean(ne > 11.345)),
            "n_updates": int(len(nis)), "mean_nis_accepted": float(nis[nis < 11.345].mean()) if (nis < 11.345).any() else None,
            "median_nis": float(np.median(nis)) if len(nis) else None,
            "frac_rejected": float(np.mean(nis >= 11.345)) if len(nis) else None,
            "final_error": r.final_error.tolist(), "duration": r.duration}


GRID = [(q, y) for q in (1.0, 2.0, 3.0, 5.0) for y in (1.0, 2.0, 3.0)]


def main(grid=GRID, seeds=(1, 2), workers: int = 6) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(q, y, s, sd) for q, y in grid for s in km.START_POSES for sd in seeds]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(one, jobs))
    summary = []
    for q, y in grid:
        rr = [r for r in rows if r["inflation"] == q and r["yaw_r_scale"] == y]
        summary.append({"q_inflation": q, "yaw_r_scale": y, "n": len(rr), "success": sum(r["status"] == "success" for r in rr),
                        "mean_nees": float(np.mean([r["mean_nees"] for r in rr])),
                        "median_nis": float(np.median([r["median_nis"] for r in rr if r["median_nis"] is not None])),
                        "frac_rejected": float(np.mean([r["frac_rejected"] for r in rr if r["frac_rejected"] is not None]))})
    (OUT / "ekf_tuning.json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=2) + "\n")
    for s in summary:
        print(s)
    ok = [s for s in summary if s["success"] == s["n"] and abs(s["mean_nees"] - 3.0) <= NEES_TOL
          and s["frac_rejected"] <= MAX_REJECTED]
    if not ok:
        print("no setting met the rule; config/ekf_tuning.json left unchanged")
        return
    best = min(ok, key=lambda s: (s["q_inflation"], s["yaw_r_scale"]))
    TUNING_JSON.write_text(json.dumps({
        "source": f"evaluation/tune_ekf.py, {len(km.START_POSES)} start poses x {len(seeds)} seeds per setting",
        "rule": f"smallest (q_inflation, yaw_r_scale) with all runs docked, |mean NEES - 3| <= {NEES_TOL} and gate "
                f"rejections <= {MAX_REJECTED:.0%}",
        "q_inflation": best["q_inflation"], "yaw_r_scale": best["yaw_r_scale"],
        "mean_nees": best["mean_nees"], "frac_rejected": best["frac_rejected"], "median_nis": best["median_nis"]},
        indent=2) + "\n")
    print("chosen:", best)


if __name__ == "__main__":
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 6)

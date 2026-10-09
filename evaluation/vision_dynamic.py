"""In-motion check of the marker measurement noise, and the inflation of the static noise model.

The static characterisation (evaluation/vision_characterisation.py) places the robot at rest. In closed-loop runs the
measurement errors are larger: the body pitches and rolls while it accelerates and turns, which tilts the camera. This
script runs docking episodes, compares every marker measurement with the measurement the TRUE pose would give
(h(x_true) with the known marker pose), and divides each error by the standard deviation the noise model predicted.
For a correct model these normalised errors have standard deviation 1. The measured standard deviation per component is
stored as "dynamic_inflation" in config/vision_noise.json and multiplies the static model at run time.

Ambiguity flips (yaw error above 15 degrees) are excluded, as in the static fit; the EKF handles them separately.
Ground truth is used here only.

Run: python -m evaluation.vision_dynamic
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from evaluation.runner import EpisodeConfig, run_episode
from src import known_map as km
from src.estimation.ekf import EKFParams
from src.tb3.interfaces import wrap
from src.vision.noise_model import NOISE_JSON

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "vision"


def h_true(x):
    m = km.MARKER_POSE_W
    c, s = np.cos(x[2]), np.sin(x[2])
    dx, dy = m[0] - x[0], m[1] - x[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(m[2] - x[2]))])


def one(args):
    start, seed = args
    # A generous process noise so that the runs complete; the measurement errors do not depend on the filter.
    r = run_episode(EpisodeConfig(start=start, seed=seed, ekf=EKFParams.calibrated(q_inflation=3.0),
                                   vision_static_noise_only=True))
    L = r.log
    out = []
    if r.status != "success":
        return r.status, out  # a robot pressed against an obstacle tilts, which is not representative
    for k in np.flatnonzero(L["detected"] == 1):
        z = np.array([L["zx"][k], L["zy"][k], L["zyaw"][k]])
        e = z - h_true(np.array([L["x"][k], L["y"][k], L["th"][k]]))
        e[2] = wrap(e[2])
        out.append([L["z_range"][k], *e, L["sx"][k], L["sy"][k], L["syaw"][k],
                    abs(L["w_odom"][k]) if k > 0 else 0.0, abs(L["v_odom"][k]) if k > 0 else 0.0])
    return r.status, out


def main(seeds=(11, 12, 13), workers: int = 6) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, sd) for s in km.START_POSES for sd in seeds]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(one, jobs))
    statuses = [st for st, _ in res]
    D = np.array([row for _, rows in res for row in rows])
    rng, E, S = D[:, 0], D[:, 1:4], D[:, 4:7]
    ok = np.abs(E[:, 2]) <= np.deg2rad(15.0)
    Z = E[ok] / S[ok]
    # Never shrink a component below the static model: the inflation is at least 1.
    infl = np.maximum(Z.std(axis=0), 1.0)
    table = []
    for lo, hi in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.6)]:
        sel = ok & (rng >= lo) & (rng < hi)
        if sel.sum() >= 5:
            table.append({"range_m": f"{lo:.1f}-{hi:.1f}", "n": int(sel.sum()),
                          "mean_mm_mm_deg": np.round(E[sel].mean(0) * [1e3, 1e3, np.rad2deg(1)], 2).tolist(),
                          "std_mm_mm_deg": np.round(E[sel].std(0) * [1e3, 1e3, np.rad2deg(1)], 2).tolist(),
                          "normalised_std": np.round((E[sel] / S[sel]).std(0), 2).tolist()})
    cfg = json.loads(NOISE_JSON.read_text())
    cfg["dynamic_inflation"] = infl.tolist()
    n_ok = sum(st == "success" for st in statuses)
    cfg["dynamic_source"] = (f"evaluation/vision_dynamic.py, {n_ok} of {len(jobs)} docking episodes successful and used, "
                             f"{int(ok.sum())} measurements ({int((~ok).sum())} ambiguity flips excluded)")
    NOISE_JSON.write_text(json.dumps(cfg, indent=2) + "\n")
    summary = {"dynamic_inflation": infl.tolist(), "normalised_std": Z.std(axis=0).tolist(),
               "episodes": dict(zip([f"{a}/{b}" for a, b in jobs], statuses)), "n": int(ok.sum()), "n_flips": int((~ok).sum()), "by_range": table}
    (OUT / "vision_dynamic.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 6)

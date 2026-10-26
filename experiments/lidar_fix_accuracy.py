"""Accuracy of the LiDAR position fix against the true pose (the LiDAR counterpart of experiments/calibrate_camera.py).

Full missions are run in random rooms; every LiDAR scan is saved with the TRUE pose at that moment. Afterwards each scan
is matched to the walls (robot_code/lidar.py, wall_fix) starting from the true pose plus a fixed offset of
(+2 cm, -2 cm, +0.6 degree), like a typical prediction error, and the fix is compared with the truth.

Reported: mean and standard deviation of the fix error, the standard deviation the fix itself predicts (from its
covariance, before the lidar_r_scale of the tuning), and their ratio. A ratio of 1 means the least-squares covariance is
right; above 1 means the fix is more confident than it should be.

Run: python -m experiments.lidar_fix_accuracy
Output: results/calibration/lidar_fix_accuracy.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import experiments.run_simulation as rs
from experiments.run_simulation import EpisodeConfig, run_episode
from experiments.true_position import true_pose
from robot_code import random_room as rl
from robot_code.data_types import wrap
from robot_code.lidar import wall_fix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "calibration"
SEEDS = (500, 501, 502, 503)
GUESS_OFFSET = np.array([0.02, -0.02, np.deg2rad(0.6)])


def main() -> None:
    caps = []
    original = rs.TB3Sim.step
    for seed in SEEDS:
        layout, _ = rl.sample_layout(seed)

        def step(self, cmd, layout=layout):
            obs = original(self, cmd)
            if obs.lidar is not None:
                caps.append((obs.lidar, true_pose(self), layout.dock_hint_y,
                             tuple(self.sensor_extrinsics()["lidar"]["t"][:2])))
            return obs

        rs.TB3Sim.step = step
        try:
            run_episode(EpisodeConfig(layout=layout, seed=seed))
        finally:
            rs.TB3Sim.step = original
    E, S, n_none = [], [], 0
    for scan, truth, hint, offset in caps:
        fix = wall_fix(scan, truth + GUESS_OFFSET, hint, offset)
        if fix is None:
            n_none += 1
            continue
        e = fix.z - truth
        e[2] = wrap(e[2])
        E.append(e)
        S.append(np.sqrt(np.diag(fix.R)))
    E, S = np.array(E), np.array(S)
    scale = np.array([1000, 1000, np.rad2deg(1)])
    out = {"seeds": list(SEEDS), "scans": len(caps), "fixes": len(E), "no_fix": n_none,
           "units": "mm, mm, deg",
           "mean_error": np.round(E.mean(0) * scale, 2).tolist(),
           "std_error": np.round(E.std(0) * scale, 2).tolist(),
           "median_predicted_std": np.round(np.median(S, 0) * scale, 2).tolist(),
           "ratio_true_to_predicted": np.round((E / S).std(0), 2).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lidar_fix_accuracy.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

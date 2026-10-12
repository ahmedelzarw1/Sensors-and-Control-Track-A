"""Measurement noise model R(d, a) for the planar marker pose, from the offline characterisation.

The coefficients and the range-bias constant are read from settings/camera_noise.json, which is written by
experiments/calibration/calibrate_camera.py from rendered images compared with ground truth. That fit is EMPIRICAL:

    sigma_i(d, a) = sqrt(c0^2 + (c1 d^2)^2) * (1 + c2 |a|),   i in {x, y, yaw}

with d the planar distance to the marker (m) and a the viewing angle (rad). The covariance is diagonal: the
correlation between components was not modelled.

The characterisation was static (robot at rest, noise-free renderer). During motion the body pitches and rolls, which
tilts the camera, so the errors are larger. Two corrections are applied:
- a floor on each standard deviation (an ASSUMPTION covering effects neither characterisation sees, such as pixel
  noise of a real camera), and
- a per-component "dynamic_inflation" factor MEASURED by experiments/calibration/camera_noise_while_driving.py, which compares measurements taken
  during docking runs with the measurement the true pose would give, and
- a further yaw scale for the correlation of the yaw errors over consecutive frames, chosen by the consistency tuning
  in experiments/calibration/tune_kalman_filter.py (settings/kalman_filter_tuning.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NOISE_JSON = Path(__file__).resolve().parents[2] / "settings" / "camera_noise.json"
TUNING_JSON = Path(__file__).resolve().parents[2] / "settings" / "kalman_filter_tuning.json"

SIGMA_FLOOR = np.array([0.005, 0.005, np.deg2rad(0.5)])  # assumption, see module docstring
INFLATION = 1.0                                            # assumption, see module docstring


class MarkerNoiseModel:
    def __init__(self, coeffs: dict, range_bias_k: float = 0.0, floor=SIGMA_FLOOR, inflation=INFLATION):
        self.c = np.array([coeffs["x"], coeffs["y"], coeffs["yaw"]], dtype=float)  # (3 components, 3 coefficients)
        self.range_bias_k = float(range_bias_k)
        self.floor = np.asarray(floor, dtype=float)
        self.inflation = np.broadcast_to(np.asarray(inflation, dtype=float), (3,)).copy()

    @classmethod
    def from_json(cls, path: Path = NOISE_JSON, static_only: bool = False, use_tuning: bool = True,
                  **kw) -> "MarkerNoiseModel":
        """static_only: the static fit only. Otherwise the measured in-motion inflation is applied and, if use_tuning,
        also the yaw scale for correlated errors chosen by experiments/calibration/tune_kalman_filter.py (settings/kalman_filter_tuning.json)."""
        cfg = json.loads(Path(path).read_text())
        if not static_only and "inflation" not in kw:
            infl = np.array(cfg.get("dynamic_inflation", [1.0, 1.0, 1.0]), dtype=float)
            if use_tuning and TUNING_JSON.exists():
                infl[2] *= json.loads(TUNING_JSON.read_text())["yaw_r_scale"]
            kw["inflation"] = infl
        return cls(cfg["coeffs"], cfg["range_bias_k_per_m"], **kw)

    def sigma(self, d: float, a: float) -> np.ndarray:
        c0, c1, c2 = self.c[:, 0], self.c[:, 1], self.c[:, 2]
        s = np.sqrt(c0 ** 2 + (c1 * d ** 2) ** 2) * (1.0 + c2 * abs(a))
        return self.inflation * np.maximum(s, self.floor)

    def covariance(self, d: float, a: float) -> np.ndarray:
        return np.diag(self.sigma(d, a) ** 2)

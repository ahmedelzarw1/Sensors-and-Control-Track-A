"""Evaluation metrics. Ground truth enters only through arguments passed in by evaluation code."""
from __future__ import annotations

import numpy as np

from src.model.unicycle import tracking_error
from src.tb3.interfaces import ReferenceTrajectory, wrap

CHI2_3_95 = 7.815  # 95% point of chi-square with 3 degrees of freedom


def tracking_metrics(true_poses: np.ndarray, ref: ReferenceTrajectory, cmds: np.ndarray | None = None) -> dict:
    """Tracking errors of the TRUE poses against the reference.

    true_poses[k] is the true pose at the same sample time as ref sample k (N x 3).
    cmds is an optional N x 2 array of commanded (v, omega) for control-effort metrics.
    """
    E = np.array([tracking_error(p, ref.pose(k)) for k, p in enumerate(true_poses)])
    out = {
        "rms_along": float(np.sqrt(np.mean(E[:, 0] ** 2))),
        "rms_cross": float(np.sqrt(np.mean(E[:, 1] ** 2))),
        "rms_heading": float(np.sqrt(np.mean(E[:, 2] ** 2))),
        "max_pos": float(np.max(np.hypot(E[:, 0], E[:, 1]))),
        "final_pos": float(np.hypot(E[-1, 0], E[-1, 1])),
        "final_heading": float(abs(E[-1, 2])),
    }
    if cmds is not None:
        fb = np.asarray(cmds) - np.column_stack([ref.v, ref.omega])[: len(cmds)]
        out["rms_fb_v"] = float(np.sqrt(np.mean(fb[:, 0] ** 2)))
        out["rms_fb_omega"] = float(np.sqrt(np.mean(fb[:, 1] ** 2)))
    return out


def nees(true_poses: np.ndarray, x_hat: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared per sample. A consistent 3-state filter averages about 3."""
    err = true_poses - x_hat
    err[:, 2] = wrap(err[:, 2])
    return np.array([float(e @ np.linalg.solve(Pk, e)) for e, Pk in zip(err, P)])

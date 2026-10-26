"""The only place in the project that reads simulator ground truth.

Everything here is for evaluating results (error metrics, plots, the true state shown
next to the estimate in the demo video). The model, controller, planner and estimator
live in robot_code/ and must never import this module.
"""
from __future__ import annotations

import numpy as np

from robot_code.simulator import TB3Sim


def true_pose(sim: TB3Sim) -> np.ndarray:
    """True (x, y, yaw) of the robot in the world frame."""
    return sim._ground_truth_pose()  # noqa: SLF001 (deliberate, evaluation-only access)

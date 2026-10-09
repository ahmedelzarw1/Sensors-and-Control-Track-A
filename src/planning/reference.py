"""Build a ReferenceTrajectory from a velocity profile.

The docking trajectory (vision part) and the obstacle-aware planner (planning part) both
end up as time-indexed (v, omega) profiles. This integrates the exact unicycle so that the pose
and velocity columns are consistent, which the feedforward term of the tracker relies on.
"""
from __future__ import annotations

import numpy as np

from src.model.unicycle import f_unicycle
from src.tb3.interfaces import ReferenceTrajectory


def integrate_reference(pose0, v, omega, Ts: float) -> ReferenceTrajectory:
    """Reference with N samples from N velocity pairs. Sample k is the pose at time k*Ts."""
    v, omega = np.asarray(v, dtype=float), np.asarray(omega, dtype=float)
    N = len(v)
    X = np.zeros((N, 3))
    X[0] = pose0
    for k in range(N - 1):
        X[k + 1] = f_unicycle(X[k], (v[k], omega[k]), Ts)
    return ReferenceTrajectory(t=Ts * np.arange(N), x=X[:, 0], y=X[:, 1], theta=X[:, 2], v=v, omega=omega)

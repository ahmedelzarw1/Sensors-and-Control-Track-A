"""LQR trajectory tracker on the linearised tracking-error model.

The controller takes only a StateEstimate and the reference. It never sees ground truth.

Three ways of choosing the gain are provided so they can be compared in the evaluation:
- "fixed":     one gain designed at a nominal operating point (v_nom, omega = 0).
- "scheduled": the discrete algebraic Riccati equation is solved at each reference sample
               (frozen-time LQR). Cheap for a 3-state model.
- "tv":        finite-horizon time-varying LQR by a backward Riccati recursion along the reference.

Weights follow Bryson's rule. The default numbers below are INITIAL values chosen for a first
working design. They are not yet tuned or validated. Record the final values and the
justification in docs/design_choices.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from src.model.unicycle import error_model, tracking_error
from src.tb3.interfaces import ControlCmd, ReferenceTrajectory, StateEstimate


def bryson(max_state: np.ndarray, max_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bryson's rule: Q_ii = 1 / max_state_i^2, R_jj = 1 / max_input_j^2."""
    return np.diag(1.0 / np.square(max_state)), np.diag(1.0 / np.square(max_input))


# Initial weights: 5 cm position error, 5 degree heading error, 5 cm/s and 0.5 rad/s of feedback.
Q_DEFAULT, R_DEFAULT = bryson(np.array([0.05, 0.05, np.deg2rad(5.0)]), np.array([0.05, 0.5]))


@dataclass(frozen=True)
class Limits:
    """Operating limits applied to the commanded body velocities.

    The default numbers are the TurtleBot3 Burger operating limits from the ROBOTIS datasheet
    (0.22 m/s, 2.84 rad/s). They are a design choice and are NOT read from the MJCF, whose only
    limit is the wheel speed range (12 rad/s). Confirm before quoting them in the report.
    """
    v_max: float = 0.22
    omega_max: float = 2.84


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete LQR gain K (u = -K x) and Riccati solution P."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def design_gain(v_r: float, w_r: float, Ts: float, Q=Q_DEFAULT, R=R_DEFAULT, v_floor: float = 0.02) -> np.ndarray:
    """LQR gain at one reference operating point.

    At v_r = 0 and w_r = 0 the linearised model is not controllable (the lateral error e_y cannot be
    influenced), so the Riccati equation has no stabilising solution. The design speed is floored at
    v_floor so that a gain always exists. The consequence for docking (the robot cannot be
    regulated to a standstill by this law) is discussed in docs/model_derivation.md.
    """
    v_design = np.sign(v_r if v_r != 0 else 1.0) * max(abs(v_r), v_floor)
    Ad, Bd = error_model(v_design, w_r, Ts)
    K, _ = dlqr(Ad, Bd, Q, R)
    return K


def tvlqr_gains(ref: ReferenceTrajectory, Ts: float, Q=Q_DEFAULT, R=R_DEFAULT, Qf=None, v_floor: float = 0.02) -> np.ndarray:
    """Finite-horizon time-varying LQR gains along a reference. Returns an array of shape (N-1, 2, 3)."""
    N = len(ref)
    P = Q.copy() if Qf is None else Qf.copy()
    Ks = np.zeros((N - 1, 2, 3))
    for k in range(N - 2, -1, -1):
        v_r = ref.v[k]
        v_design = np.sign(v_r if v_r != 0 else 1.0) * max(abs(v_r), v_floor)
        Ad, Bd = error_model(v_design, ref.omega[k], Ts)
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        P = Q + Ad.T @ P @ (Ad - Bd @ K)
        P = 0.5 * (P + P.T)
        Ks[k] = K
    return Ks


class LQRTracker:
    """Feedforward plus LQR feedback on the estimated tracking error."""

    def __init__(self, Ts: float, mode: str = "scheduled", Q=Q_DEFAULT, R=R_DEFAULT,
                 limits: Limits = Limits(), v_nom: float = 0.1, Qf=None):
        if mode not in ("fixed", "scheduled", "tv"):
            raise ValueError("mode must be 'fixed', 'scheduled' or 'tv'")
        self.Ts, self.mode, self.Q, self.R, self.limits, self.Qf = Ts, mode, Q, R, limits, Qf
        self._K_fixed = design_gain(v_nom, 0.0, Ts, Q, R) if mode == "fixed" else None
        self._K_tv = None
        self._tv_for = None

    def gain(self, ref: ReferenceTrajectory, k: int) -> np.ndarray:
        if self.mode == "fixed":
            return self._K_fixed
        if self.mode == "scheduled":
            kk = min(k, len(ref) - 1)
            return design_gain(ref.v[kk], ref.omega[kk], self.Ts, self.Q, self.R)
        if self._tv_for is not ref:
            self._K_tv = tvlqr_gains(ref, self.Ts, self.Q, self.R, self.Qf)
            self._tv_for = ref
        return self._K_tv[min(k, len(self._K_tv) - 1)]

    def command(self, est: StateEstimate, ref: ReferenceTrajectory, k: int) -> ControlCmd:
        e = tracking_error(est.x_hat, ref.pose(k))
        du = -self.gain(ref, k) @ e
        u_ref = ref.vel(k)
        v = float(np.clip(u_ref[0] + du[0], -self.limits.v_max, self.limits.v_max))
        w = float(np.clip(u_ref[1] + du[1], -self.limits.omega_max, self.limits.omega_max))
        return ControlCmd(v=v, omega=w)

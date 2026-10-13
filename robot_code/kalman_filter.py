"""Extended Kalman filter: wheel odometry prediction, ArUco marker pose update.

Baseline written for the EKF teammate to take over. Implements the Estimator protocol in robot_code/data_types.py.
How the noise values were measured and tuned: README.md (How it works).

State   x = [x, y, theta], the pose of the base frame B in W.
Predict x+ = f(x, u_odom)            (exact unicycle over one period, robot_code/unicycle_model.py)
        P+ = F P F^T + G Qu G^T + Qa
        F = df/dx, G = df/du. Qu = diag(sigma_v^2, sigma_omega^2) is the odometry noise. Qa is extra noise per step
        for what the odometry model does not capture (wheel slip, the transient after command changes, wheel radius
        error). Values: EKFParams below, measured by experiments/calibrate_wheels.py.
Update  z = base_T_marker = [x_M, y_M, yaw_M] in B, from the camera (robot_code/vision).
        h(x) = [R(theta)^T (m_xy - p), wrap(m_yaw - theta)] with the known marker pose m in W (robot_code/room_map.py).
        H = dh/dx (H_marker_pose in robot_code/unicycle_model.py). Innovation nu = z - h(x) with the yaw wrapped.
        Using the full pose of the marker (not only range and bearing) makes the whole pose observable from a single
        marker (robot_code/unicycle_model.py, H_marker_pose, and tests/test_movement_maths.py).
Gate    NIS = nu^T S^-1 nu, S = H P H^T + R. The update is applied only if NIS < chi2(3 dof, gate_prob). This rejects
        the planar pose ambiguity flips of the square marker and any wrong detection.
Update form: Joseph form, which keeps P symmetric and positive definite.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import chi2

from robot_code import room_map as km
from robot_code.unicycle_model import F_unicycle, G_unicycle, H_marker_pose, f_unicycle
from robot_code.data_types import MarkerMeasurement, Odometry, StateEstimate, wrap


CALIBRATION_JSON = Path(__file__).resolve().parents[1] / "settings" / "wheel_calibration.json"
TUNING_JSON = Path(__file__).resolve().parents[1] / "settings" / "kalman_filter_tuning.json"


@dataclass(frozen=True)
class EKFParams:
    # Odometry noise: equal to the simulator's default odometry noise, which is itself an assumption (INTERFACES.md).
    sigma_v: float = 0.005            # m/s
    sigma_omega: float = 0.01         # rad/s
    # Odometry calibration: the measured v and omega are multiplied by these before prediction.
    scale_v: float = 1.0
    scale_omega: float = 1.0
    # Extra process noise per 0.1 s step (standard deviations): a constant part and parts proportional to the distance
    # and angle moved in the step. Measured values come from settings/wheel_calibration.json (see calibrated()).
    q_xy: float = 0.002               # m
    q_theta: float = 0.004            # rad
    k_dist: float = 0.02              # m of position noise std per m travelled
    k_turn: float = 0.02              # rad of heading noise std per rad turned
    q_inflation: float = 1.0          # multiplies the extra noise std (chosen by experiments/tune_kalman_filter.py)
    gate_prob: float = 0.99           # chi-square gate probability, 3 degrees of freedom
    # Re-initialisation after repeated rejections (recovery from a lost filter). See reinit logic below.
    reinit_after: int = 8             # consecutive gated-out measurements that agree with each other
    reinit_max_range: float = 1.6     # m, only near measurements (no ambiguity flips seen below 1.5 m)

    @classmethod
    def calibrated(cls, path: Path = CALIBRATION_JSON, use_tuning: bool = True, **overrides) -> "EKFParams":
        """Parameters from the offline odometry calibration (experiments/calibrate_wheels.py) and, if use_tuning,
        the process-noise inflation chosen by the consistency tuning (experiments/tune_kalman_filter.py)."""
        cal = json.loads(Path(path).read_text())
        q = cal["extra_noise_per_step"]
        kw = dict(scale_v=cal["scale_v"], scale_omega=cal["scale_omega"], q_xy=q["q_xy_m"], q_theta=q["q_theta_rad"],
                  k_dist=q["k_dist"], k_turn=q["k_turn"])
        if use_tuning and TUNING_JSON.exists():
            kw["q_inflation"] = json.loads(TUNING_JSON.read_text())["q_inflation"]
        kw.update(overrides)
        return cls(**kw)


@dataclass
class UpdateLog:
    t: float
    nis: float
    accepted: bool
    innovation: np.ndarray
    S: np.ndarray


class MarkerEKF:
    def __init__(self, x0, P0, params: EKFParams = EKFParams(), marker_pose_w: np.ndarray = km.MARKER_POSE_W):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.p = params
        self.m = np.asarray(marker_pose_w, dtype=float)
        self.t = 0.0
        self.gate = float(chi2.ppf(params.gate_prob, 3))
        self.updates: list[UpdateLog] = []
        self.reinit_events: list[float] = []
        self._rejected: list[np.ndarray] = []   # poses implied by consecutive rejected measurements

    # ---------------------------------------------------------------- predict
    def predict(self, odom: Odometry) -> None:
        u = np.array([self.p.scale_v * odom.v, self.p.scale_omega * odom.omega])
        F = F_unicycle(self.x, u, odom.dt)
        G = G_unicycle(self.x, u, odom.dt)
        Qu = np.diag([self.p.sigma_v ** 2, self.p.sigma_omega ** 2])
        dist, turn = abs(odom.v) * odom.dt, abs(odom.omega) * odom.dt
        sxy = self.p.q_inflation * np.hypot(self.p.q_xy, self.p.k_dist * dist)
        sth = self.p.q_inflation * np.hypot(self.p.q_theta, self.p.k_turn * turn)
        Qa = np.diag([sxy ** 2, sxy ** 2, sth ** 2])
        self.x = f_unicycle(self.x, u, odom.dt)
        self.P = F @ self.P @ F.T + G @ Qu @ G.T + Qa
        self.P = 0.5 * (self.P + self.P.T)
        self.t = odom.t

    # ---------------------------------------------------------------- update
    def h(self, x: np.ndarray) -> np.ndarray:
        c, s = np.cos(x[2]), np.sin(x[2])
        dx, dy = self.m[0] - x[0], self.m[1] - x[1]
        return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(self.m[2] - x[2]))])

    def pose_from_measurement(self, z: np.ndarray) -> np.ndarray:
        """Robot pose in W implied by one marker measurement (inverse of h)."""
        th = float(wrap(self.m[2] - z[2]))
        c, s = np.cos(th), np.sin(th)
        return np.array([self.m[0] - (c * z[0] - s * z[1]), self.m[1] - (s * z[0] + c * z[1]), th])

    def update_marker(self, meas: MarkerMeasurement) -> bool:
        """Apply one marker pose measurement. Returns True if it passed the gate and was used."""
        if meas.base_T_marker is None or meas.R is None:
            return False
        H = H_marker_pose(self.x, self.m)
        S = H @ self.P @ H.T + meas.R
        # When the vision part reports both solutions of the square-marker ambiguity, use the one closer to the
        # prediction (smaller NIS). The gate below still applies to it.
        cands = [meas.base_T_marker] + ([meas.base_T_marker_alt] if meas.base_T_marker_alt is not None else [])
        best = None
        for zc in cands:
            zc = np.asarray(zc, dtype=float)
            nu_c = zc - self.h(self.x)
            nu_c[2] = wrap(nu_c[2])
            nis_c = float(nu_c @ np.linalg.solve(S, nu_c))
            if best is None or nis_c < best[0]:
                best = (nis_c, nu_c, zc)
        nis, nu, z = best
        ok = nis < self.gate
        self.updates.append(UpdateLog(meas.t, nis, ok, nu, S))
        if ok:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ nu
            self.x[2] = wrap(self.x[2])
            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ meas.R @ K.T
            self.P = 0.5 * (self.P + self.P.T)
            self._rejected.clear()
        else:
            self._maybe_reinitialise(z, meas)
        return ok

    def _maybe_reinitialise(self, z: np.ndarray, meas: MarkerMeasurement) -> None:
        """Recovery: if several consecutive near measurements are rejected but agree with each other, the filter is
        assumed lost (for example after a collision or wheel slip), and the pose is re-initialised from them."""
        if np.hypot(z[0], z[1]) > self.p.reinit_max_range:
            self._rejected.clear()
            return
        # Offset of the implied pose from the prediction. The prediction follows the odometry, so if the filter is lost
        # by a fixed amount the offsets of consecutive measurements agree even while the robot moves.
        implied = self.pose_from_measurement(z)
        off = implied - self.x
        off[2] = wrap(off[2])
        self._rejected.append(off)
        if len(self._rejected) < self.p.reinit_after:
            return
        offs = np.array(self._rejected[-self.p.reinit_after:])
        if np.ptp(offs[:, :2], axis=0).max() < 0.05 and np.ptp(offs[:, 2]) < np.deg2rad(5.0):
            J = self._inverse_jacobian(z)
            self.x = implied
            self.P = J @ meas.R @ J.T + np.diag([0.02, 0.02, np.deg2rad(2.0)]) ** 2
            self.reinit_events.append(meas.t)
            self._rejected.clear()

    def _inverse_jacobian(self, z: np.ndarray) -> np.ndarray:
        eps = 1e-6
        J = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3); d[i] = eps
            a, b = self.pose_from_measurement(z + d), self.pose_from_measurement(z - d)
            diff = a - b
            diff[2] = wrap(diff[2])
            J[:, i] = diff / (2 * eps)
        return J

    # ---------------------------------------------------------------- output
    def estimate(self) -> StateEstimate:
        return StateEstimate(t=self.t, x_hat=self.x.copy(), P=self.P.copy())

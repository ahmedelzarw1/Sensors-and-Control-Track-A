"""Docking by visual servoing: the final approach from the approach point into the charging station.

The robot is only told roughly where the station is (within 0.20 m). So the final approach is not done in the room
frame at all: it is done in the MARKER frame M, from what the camera sees.

1. Prior (marker_frame_prior). At the approach point the robot turns to face the given station position, and its
   room-frame estimate is converted into a first guess of its pose relative to the marker, with the uncertainty of the
   given station position added along the wall.
2. Docking EKF (robot_code/kalman_filter.py, in M). Wheel odometry predicts, and every camera measurement of the marker
   corrects the robot pose relative to the marker. The chi-square gate rejects the square-marker ambiguity flips.
3. Reference (docking_reference), built in M from the docking EKF estimate. A cubic Bezier curve joins the marker
   normal line at a join point JOIN_DISTANCE in front of the dock pose, tangent to the normal, then a straight line runs
   to the dock pose and continues past it by EXTEND, driven at V_DOCK. The reference never slows to zero: the tracking
   model is not controllable at zero reference speed (robot_code/unicycle_model.py, ctrb), so the motion is ended by the
   stop rule.
4. Control. The LQR tracker (robot_code/lqr_controller.py) follows the reference using the docking EKF estimate, so
   the command comes from the camera's view of the marker: position-based visual servoing.
5. Stop rule. Stop (command zero) when the ESTIMATED distance still to go along the approach direction falls below half
   a control step of travel.

The dock pose in M is DOCK_POSE_M = (DOCK_STANDOFF, 0, pi): 0.25 m out along the marker normal, facing the marker.
Success (evaluation only, measured on the TRUE final pose against the TRUE dock): see DockTolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from robot_code import room_map as km
from robot_code.path_planning import DensePath, ProfileLimits, reference_from_path, speed_profile
from robot_code.data_types import ControlCmd, ReferenceTrajectory, StateEstimate, wrap

JOIN_DISTANCE = 0.30      # m before the dock, where the curve meets the normal line (assumption)
EXTEND = 0.30             # m the reference continues past the dock
V_DOCK = 0.06             # m/s final approach speed (assumption, slow so the stop overshoot is small)
DOCK_LIMITS = ProfileLimits(v_cruise=0.10, omega_max=0.8, a_max=0.10, v_min=0.03)


@dataclass(frozen=True)
class DockTolerance:
    """Docking success, checked on the TRUE final pose in experiments. ASSUMPTIONS: a charging dock with spring
    contacts about 4 cm wide is taken to need the robot centred within 2 cm laterally, within 3 cm along the approach
    (spring travel) and aligned within 5 degrees."""
    lateral: float = 0.02
    longitudinal: float = 0.03
    heading: float = np.deg2rad(5.0)


def dock_frame_error(pose: np.ndarray, dock: np.ndarray = km.DOCK_POSE_M) -> np.ndarray:
    """(longitudinal, lateral, heading) error of a pose relative to the dock pose, in the dock frame.
    Longitudinal is positive when the robot has stopped short of the dock."""
    c, s = np.cos(dock[2]), np.sin(dock[2])
    dx, dy = dock[0] - pose[0], dock[1] - pose[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(dock[2] - pose[2]))])


def remaining_distance(est: StateEstimate, dock: np.ndarray = km.DOCK_POSE_M) -> float:
    return float(dock_frame_error(est.x_hat, dock)[0])


def should_stop(est: StateEstimate, v_cmd: float, Ts: float, dock: np.ndarray = km.DOCK_POSE_M) -> bool:
    return remaining_distance(est, dock) <= 0.5 * max(v_cmd, 0.0) * Ts


def docking_path(start_pose: np.ndarray, dock: np.ndarray = km.DOCK_POSE_M, ds: float = 0.005) -> DensePath:
    """Cubic Bezier from the start pose to the join point, then the straight run to the dock."""
    u = np.array([np.cos(dock[2]), np.sin(dock[2])])
    join = dock[:2] - JOIN_DISTANCE * u
    p0 = start_pose[:2]
    t0 = np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
    L = np.linalg.norm(join - p0)
    if L < 0.05 or (join - p0) @ u <= 0.0:
        # Already at or past the join point: drive the straight line from the projection on the normal.
        pts = [p0, dock[:2]]
        bez = np.array(pts)
    else:
        c1, c2 = p0 + t0 * L / 3, join - u * L / 3
        tt = np.linspace(0, 1, max(20, int(L / ds)))[:, None]
        bez = (1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * c1 + 3 * (1 - tt) * tt ** 2 * c2 + tt ** 3 * join
        n = max(2, int(JOIN_DISTANCE / ds))
        straight = join + np.linspace(0, 1, n)[1:, None] * (dock[:2] - join)
        bez = np.vstack([bez, straight])
    seg = np.diff(bez, axis=0)
    keep = np.r_[True, np.hypot(seg[:, 0], seg[:, 1]) > 1e-9]
    bez = bez[keep]
    seg = np.diff(bez, axis=0)
    th = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    th = np.append(th, th[-1])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    kappa = np.gradient(th, s, edge_order=1) if len(s) > 2 else np.zeros_like(s)
    return DensePath(s, bez[:, 0], bez[:, 1], th, kappa)


def docking_reference(est_pose: np.ndarray, Ts: float, v_start: float = 0.05,
                      dock: np.ndarray = km.DOCK_POSE_M) -> ReferenceTrajectory:
    path = docking_path(np.asarray(est_pose, float), dock)
    v = speed_profile(path, DOCK_LIMITS, v_start, V_DOCK)
    # Keep the final straight at the approach speed.
    on_straight = path.s >= path.s[-1] - JOIN_DISTANCE
    v[on_straight] = np.minimum(v[on_straight], V_DOCK)
    return reference_from_path(path, v, Ts, extend=EXTEND, v_extend=V_DOCK)


def join_point(dock: np.ndarray = km.DOCK_POSE_M) -> np.ndarray:
    """Where the docking curve meets the marker normal line."""
    return dock[:2] - JOIN_DISTANCE * np.array([np.cos(dock[2]), np.sin(dock[2])])


def marker_frame_prior(x_w: np.ndarray, P_w: np.ndarray, dock_hint_y: float) -> tuple[np.ndarray, np.ndarray]:
    """First guess of the robot pose in the marker frame M, from the room-frame estimate (x_w, P_w) and the given
    station position. The marker is assumed at the given position, so its error along the wall (uniform within
    DOCK_HINT_ERROR, standard deviation DOCK_HINT_ERROR / sqrt(3)) is added to the covariance.

        p_M = R(m_yaw)^T (p_W - m_xy),   theta_M = theta_W - m_yaw
    """
    m = km.marker_pose(dock_hint_y)
    c, s = np.cos(m[2]), np.sin(m[2])
    Rt = np.array([[c, s], [-s, c]])
    x_m = np.r_[Rt @ (x_w[:2] - m[:2]), wrap(x_w[2] - m[2])]
    J = np.eye(3)
    J[:2, :2] = Rt
    dm = -Rt @ np.array([0.0, 1.0])                  # change of p_M per metre of marker shift along the wall (y)
    sigma_hint = km.DOCK_HINT_ERROR / np.sqrt(3.0)
    P_m = J @ P_w @ J.T
    P_m[:2, :2] += sigma_hint ** 2 * np.outer(dm, dm)
    return x_m, P_m


class MarkerSearch:
    """Turn in place to bring the marker into view, using only the estimate and where the marker should be.

    1. Turn towards the bearing at which the estimate says the marker is.
    2. If it is still not detected, sweep +-SWEEP about that bearing, then make a full turn.
    The caller stops the search as soon as a marker measurement is accepted by the filter.
    """
    SWEEP = np.deg2rad(45.0)
    OMEGA = 0.5   # rad/s (assumption: slow enough that the image does not blur in a real camera)

    def __init__(self):
        self._stage = 0
        self._target: Optional[float] = None
        self._turned = 0.0

    def command(self, est: StateEstimate, marker_xy, Ts: float) -> ControlCmd:
        """marker_xy: where the marker should be, in the frame of the estimate."""
        x, y, th = est.x_hat
        bearing_w = float(np.arctan2(marker_xy[1] - y, marker_xy[0] - x))
        targets = [bearing_w, bearing_w + self.SWEEP, bearing_w - self.SWEEP]
        if self._stage < len(targets):
            err = float(wrap(targets[self._stage] - th))
            if abs(err) < np.deg2rad(3.0):
                self._stage += 1
                return ControlCmd(0.0, 0.0)
            return ControlCmd(0.0, float(np.clip(1.5 * err, -self.OMEGA, self.OMEGA)))
        # Full turn as the last resort.
        self._turned += self.OMEGA * Ts
        if self._turned > 2 * np.pi:
            self._stage, self._turned = 0, 0.0
        return ControlCmd(0.0, self.OMEGA)

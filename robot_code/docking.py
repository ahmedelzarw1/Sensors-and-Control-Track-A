"""Docking: the reference from the pre-dock area into the charging station, the stop rule and the marker search.

Baseline written for the vision and docking teammate to take over.

Geometry (robot_code/room_map.py). The marker normal points into the room. The dock pose is DOCK_STANDOFF in front of the
marker face, facing the marker. The approach direction u is the dock heading.

Reference. From the current ESTIMATED pose a cubic Bezier curve joins the marker normal line at a join point
JOIN_DISTANCE in front of the dock, tangent to the normal, then a straight line runs to the dock and continues past
it by EXTEND. The final straight is driven at V_DOCK. The reference never slows to zero: the tracking model is not
controllable at zero reference speed (robot_code/unicycle_model.py, ctrb), so the motion is ended by the stop rule.

Stop rule. Stop (command zero) when the ESTIMATED distance still to go along u, (p_dock - p_hat) . u, falls below
half a control step of travel. Ground truth is not used.

Success (evaluation only, measured on the TRUE final pose): see DockTolerance.
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


def dock_frame_error(pose: np.ndarray, dock: np.ndarray = km.DOCK_POSE) -> np.ndarray:
    """(longitudinal, lateral, heading) error of a pose relative to the dock pose, in the dock frame.
    Longitudinal is positive when the robot has stopped short of the dock."""
    c, s = np.cos(dock[2]), np.sin(dock[2])
    dx, dy = dock[0] - pose[0], dock[1] - pose[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(dock[2] - pose[2]))])


def remaining_distance(est: StateEstimate, dock: np.ndarray = km.DOCK_POSE) -> float:
    return float(dock_frame_error(est.x_hat, dock)[0])


def should_stop(est: StateEstimate, v_cmd: float, Ts: float, dock: np.ndarray = km.DOCK_POSE) -> bool:
    return remaining_distance(est, dock) <= 0.5 * max(v_cmd, 0.0) * Ts


def docking_path(start_pose: np.ndarray, dock: np.ndarray = km.DOCK_POSE, ds: float = 0.005) -> DensePath:
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
                      dock: np.ndarray = km.DOCK_POSE) -> ReferenceTrajectory:
    path = docking_path(np.asarray(est_pose, float), dock)
    v = speed_profile(path, DOCK_LIMITS, v_start, V_DOCK)
    # Keep the final straight at the approach speed.
    on_straight = path.s >= path.s[-1] - JOIN_DISTANCE
    v[on_straight] = np.minimum(v[on_straight], V_DOCK)
    return reference_from_path(path, v, Ts, extend=EXTEND, v_extend=V_DOCK)


class MarkerSearch:
    """Turn in place to bring the marker into view, using only the estimate and the known marker position.

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

    def command(self, est: StateEstimate, Ts: float) -> ControlCmd:
        x, y, th = est.x_hat
        bearing_w = float(np.arctan2(km.MARKER_POSE_W[1] - y, km.MARKER_POSE_W[0] - x))
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

"""Shared data types and protocols for the Track A pipeline.

Everything the three subsystems exchange is defined here, so that the estimator,
the docking logic and the controller can be developed independently.

Conventions (the frame chain from camera to robot is in robot_code/vision/marker_detection.py)
- SI units. Angles in radians, wrapped to [-pi, pi).
- World frame W: MuJoCo world frame, z up. Planar pose is (x, y, theta) of the
  robot base frame B in W, theta is yaw about +z, counter-clockwise positive.
- Body frame B: body "base" of the ROBOTIS model, origin on the ground below the wheel axle centre,
  x forward, y left, z up.
- Commands are body velocities: v (m/s, forward) and omega (rad/s, CCW positive).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


def wrap(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class ControlCmd:
    """Body velocity command sent to the platform."""
    v: float
    omega: float


@dataclass(frozen=True)
class Odometry:
    """Wheel-encoder odometry over the last control period.

    v and omega are body velocities computed from the wheel encoder increments.
    They are measurements, so they include noise.
    """
    t: float
    v: float
    omega: float
    dt: float


@dataclass(frozen=True)
class LidarScan:
    """One 2D LiDAR scan in the sensor frame L.

    angles: beam angles in the sensor frame, radians, CCW from the sensor x axis.
    ranges: metres. NaN where there was no valid return (out of range or no hit).
    """
    t: float
    angles: np.ndarray
    ranges: np.ndarray


@dataclass(frozen=True)
class MarkerMeasurement:
    """Proposed camera measurement of an ArUco marker (produced by the vision part).

    Either or both representations may be provided.
    range_bearing: (range [m], bearing [rad]) of the marker centre in the base frame B.
    base_T_marker: planar pose (x, y, yaw) of the marker in the base frame B.
    R: measurement covariance matching whichever representation is used.
    base_T_marker_alt: the other solution of the square-marker pose ambiguity, given only when the two solutions fit
        the image almost equally well. The estimator may use whichever agrees with its prediction (added in Stage 1,
        see INTERFACES.md).
    """
    t: float
    marker_id: int
    range_bearing: Optional[np.ndarray] = None
    base_T_marker: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    base_T_marker_alt: Optional[np.ndarray] = None


@dataclass(frozen=True)
class StateEstimate:
    """Estimated planar pose in W with covariance. This is what the controller consumes."""
    t: float
    x_hat: np.ndarray  # shape (3,): x, y, theta
    P: np.ndarray      # shape (3, 3)


@dataclass
class ReferenceTrajectory:
    """Time-indexed reference for the tracker, sampled at the control period Ts.

    All arrays have length N. v and omega are the reference body velocities
    (feedforward). theta must be consistent with the path tangent.
    """
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    def pose(self, k: int) -> np.ndarray:
        k = min(max(k, 0), len(self) - 1)
        return np.array([self.x[k], self.y[k], self.theta[k]])

    def vel(self, k: int) -> np.ndarray:
        k = min(max(k, 0), len(self) - 1)
        return np.array([self.v[k], self.omega[k]])


@dataclass
class Observation:
    """Everything the platform returns after one control period."""
    odom: Odometry
    lidar: Optional[LidarScan] = None  # present only on steps where a scan completes
    extras: dict = field(default_factory=dict)


class Estimator(Protocol):
    """Contract for the EKF (teammates)."""

    def predict(self, odom: Odometry) -> None: ...

    def update_marker(self, meas: MarkerMeasurement) -> None: ...

    def estimate(self) -> StateEstimate: ...


class Controller(Protocol):
    """Contract for the tracker (mine). It sees only the estimate and the reference."""

    def command(self, est: StateEstimate, ref: ReferenceTrajectory, k: int) -> ControlCmd: ...

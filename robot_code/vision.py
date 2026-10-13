"""Vision: from a camera image to a marker measurement the Kalman filter can use.

Three parts, in the order they are used:
1. Camera maths: intrinsics from the field of view, and the MuJoCo to OpenCV axis conversion.
2. Marker detection: ArUco detection, pose by solvePnP, conversion into the robot frame (MarkerSensor).
3. Marker noise: the measurement covariance R as a function of distance and viewing angle (measured).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
import cv2
from robot_code import room_map as km
from robot_code.data_types import MarkerMeasurement
import json
from pathlib import Path


# ==============================================================================================================
# 1. CAMERA MATHS
# Pinhole model of the simulated robot camera and the MuJoCo to OpenCV axis conversion.
#
# The resolution and vertical field of view are set explicitly in the generated scene (commands/build_mujoco_files.py
# writes them into mujoco_files/room.xml), so the intrinsics follow from them exactly:
#
#     fy = (H / 2) / tan(fovy / 2),   fx = fy (MuJoCo renders square pixels),
#     cx = (W - 1) / 2,   cy = (H - 1) / 2   (OpenCV puts pixel centres on integer coordinates)
#
# There is no lens distortion in the renderer, so the distortion coefficients are zero. The intrinsics are checked
# indirectly by the vision characterisation (experiments/calibrate_camera.py): after the range-bias correction
# the mean pose error is below 1 mm at short range. A dedicated projection test is still to be written.
#
# Axis conventions:
# - MuJoCo camera frame: x to the image right, y up, looking along -z.
# - OpenCV camera frame: x to the image right, y down, looking along +z.
# So R_mujoco_from_opencv = diag(1, -1, -1).
# ==============================================================================================================

# Chosen so that the 0.15 m marker is detected from 0.2 m (docked) to 3.6 m (measured, README.md, How it works).
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOVY_DEG = 60.0

R_MJ_FROM_CV = np.diag([1.0, -1.0, -1.0])


def intrinsics(width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT, fovy_deg: float = CAMERA_FOVY_DEG) -> np.ndarray:
    """Camera matrix K of the ideal pinhole that MuJoCo renders."""
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
    return np.array([[fy, 0.0, 0.5 * (width - 1)],
                     [0.0, fy, 0.5 * (height - 1)],
                     [0.0, 0.0, 1.0]])


def horizontal_fov(width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT, fovy_deg: float = CAMERA_FOVY_DEG) -> float:
    """Horizontal field of view in radians."""
    return 2.0 * np.arctan(width / height * np.tan(0.5 * np.deg2rad(fovy_deg)))


DIST_COEFFS = np.zeros(5)


# ==============================================================================================================
# 2. MARKER DETECTION
# ArUco detection, marker pose by solvePnP, and conversion to a MarkerMeasurement in the base frame B.
#
# Chain from image to measurement (the frames used across the project):
# 1. detectMarkers gives the four image corners of marker id 0 (sub-pixel refined).
# 2. solvePnP with SOLVEPNP_IPPE_SQUARE gives the marker pose in the OpenCV camera frame. The square marker has a
#    two-fold pose ambiguity, so both IPPE solutions are computed and the one with the lower reprojection error is kept.
# 3. OpenCV camera frame to MuJoCo camera frame: R_MJ_FROM_CV = diag(1, -1, -1).
# 4. MuJoCo camera frame to base frame B: the camera extrinsics from TB3Sim.sensor_extrinsics().
# 5. OpenCV marker frame (x right, y up, z out of the face) to the project marker frame M (x out of the face, y right,
#    z up): x_M = z_cv, y_M = x_cv, z_M = y_cv.
# 6. Range-bias correction from the calibration (see to_base_frame).
# 7. Planar projection: base_T_marker = (x, y, yaw) where yaw is the direction of the marker normal x_M in B.
# ==============================================================================================================

# Second IPPE solution passed on when its reprojection error is below this multiple of the best one. In the vision
# characterisation, flipped detections had a median ratio of 1.4 and correct ones a median of 30.
AMBIGUITY_RATIO = 3.0

# Columns are the M axes expressed in the OpenCV marker frame.
R_CVMARKER_FROM_M = np.array([[0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0],
                              [1.0, 0.0, 0.0]])


def detector_parameters() -> "cv2.aruco.DetectorParameters":
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.01  # allow small markers far away (default 0.03 of the image size)
    return p


@dataclass(frozen=True)
class Detection:
    marker_id: int
    corners: np.ndarray        # (4, 2) pixels, ArUco order: top-left, top-right, bottom-right, bottom-left
    R_cam_marker: np.ndarray   # (3, 3) OpenCV marker frame in OpenCV camera frame
    t_cam_marker: np.ndarray   # (3,) metres, OpenCV camera frame
    reproj_err: float          # RMS pixels of the chosen solution
    reproj_err_alt: float      # RMS pixels of the other IPPE solution (ambiguity indicator)
    R_cam_marker_alt: Optional[np.ndarray] = None
    t_cam_marker_alt: Optional[np.ndarray] = None

    @property
    def ambiguous(self) -> bool:
        """True when the second IPPE solution explains the image almost as well as the first."""
        return self.t_cam_marker_alt is not None and self.reproj_err_alt < AMBIGUITY_RATIO * max(self.reproj_err, 1e-3)

    def alternative(self) -> "Detection":
        return Detection(self.marker_id, self.corners, self.R_cam_marker_alt, self.t_cam_marker_alt,
                         self.reproj_err_alt, self.reproj_err)


@dataclass(frozen=True)
class PlanarMarker:
    """Marker pose in B with the quantities the noise model needs."""
    base_T_marker: np.ndarray  # (x, y, yaw) of M in B
    range: float               # planar distance from the base frame origin to the marker centre, m
    bearing: float             # direction of the marker centre in B, rad
    view_angle: float          # angle between the marker normal and the line from the marker to the robot, rad


class ArucoDetector:
    def __init__(self, K: Optional[np.ndarray] = None, dist: np.ndarray = DIST_COEFFS,
                 marker_size: float = km.MARKER_SIZE, dictionary: str = km.ARUCO_DICT):
        self.K = intrinsics() if K is None else K
        self.dist = dist
        self.size = marker_size
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        self._det = cv2.aruco.ArucoDetector(d, detector_parameters())
        s = marker_size / 2.0
        # Object points in the OpenCV marker frame, in the corner order required by SOLVEPNP_IPPE_SQUARE.
        self._obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._det.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            img_pts = c.reshape(4, 2).astype(np.float64)
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self._obj, img_pts, self.K, self.dist,
                                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n == 0:
                continue
            errs = np.asarray(errs).ravel()
            best = int(np.argmin(errs))
            R, _ = cv2.Rodrigues(rvecs[best])
            if n > 1:
                R2, _ = cv2.Rodrigues(rvecs[1 - best])
                out.append(Detection(int(i), img_pts, R, np.asarray(tvecs[best]).ravel(), float(errs[best]),
                                     float(errs[1 - best]), R2, np.asarray(tvecs[1 - best]).ravel()))
            else:
                out.append(Detection(int(i), img_pts, R, np.asarray(tvecs[best]).ravel(), float(errs[best]),
                                     float("inf")))
        return out


def to_base_frame(det: Detection, cam_extrinsics: dict, range_bias_k: float = 0.0) -> PlanarMarker:
    """Planar marker pose in B from a detection and the camera extrinsics {"t": (3,), "R": (3,3) MuJoCo camera in B}.

    range_bias_k corrects the calibrated range bias: the camera-to-marker distance r is replaced by r - k r^2 along
    the same ray (fitted by experiments/calibrate_camera.py).
    """
    R_B_cv = cam_extrinsics["R"] @ R_MJ_FROM_CV
    t_cam = det.t_cam_marker * (1.0 - range_bias_k * float(np.linalg.norm(det.t_cam_marker)))
    t_B = cam_extrinsics["t"] + R_B_cv @ t_cam
    R_B_M = R_B_cv @ det.R_cam_marker @ R_CVMARKER_FROM_M
    normal = R_B_M[:, 0]
    yaw = float(np.arctan2(normal[1], normal[0]))
    rng = float(np.hypot(t_B[0], t_B[1]))
    bearing = float(np.arctan2(t_B[1], t_B[0]))
    # Angle between the outward normal and the direction from the marker to the robot origin (planar).
    to_robot = -t_B[:2] / max(rng, 1e-9)
    n2 = normal[:2] / max(np.linalg.norm(normal[:2]), 1e-9)
    view = float(np.arctan2(n2[0] * to_robot[1] - n2[1] * to_robot[0], n2 @ to_robot))
    return PlanarMarker(np.array([t_B[0], t_B[1], yaw]), rng, bearing, view)


class MarkerSensor:
    """Image in, MarkerMeasurement out. Holds the detector, the camera extrinsics and the noise model."""

    def __init__(self, cam_extrinsics: dict, noise_model, marker_id: int = km.MARKER_ID,
                 detector: Optional[ArucoDetector] = None):
        self.extr = cam_extrinsics
        self.noise = noise_model
        self.marker_id = marker_id
        self.detector = detector if detector is not None else ArucoDetector()
        self.last_detection: Optional[Detection] = None
        self.last_planar: Optional[PlanarMarker] = None

    def measure(self, image_rgb: np.ndarray, t: float) -> Optional[MarkerMeasurement]:
        self.last_detection, self.last_planar = None, None
        for det in self.detector.detect(image_rgb):
            if det.marker_id != self.marker_id:
                continue
            pm = to_base_frame(det, self.extr, self.noise.range_bias_k)
            self.last_detection, self.last_planar = det, pm
            R = self.noise.covariance(pm.range, pm.view_angle)
            alt = (to_base_frame(det.alternative(), self.extr, self.noise.range_bias_k).base_T_marker
                   if det.ambiguous else None)
            return MarkerMeasurement(t=t, marker_id=det.marker_id,
                                     range_bearing=np.array([pm.range, pm.bearing]),
                                     base_T_marker=pm.base_T_marker, R=R, base_T_marker_alt=alt)
        return None


# ==============================================================================================================
# 3. MARKER NOISE
# Measurement noise model R(d, a) for the planar marker pose, from the offline characterisation.
#
# The coefficients and the range-bias constant are read from settings/camera_noise.json, which is written by
# experiments/calibrate_camera.py from rendered images compared with ground truth. That fit is EMPIRICAL:
#
#     sigma_i(d, a) = sqrt(c0^2 + (c1 d^2)^2) * (1 + c2 |a|),   i in {x, y, yaw}
#
# with d the planar distance to the marker (m) and a the viewing angle (rad). The covariance is diagonal: the
# correlation between components was not modelled.
#
# The characterisation was static (robot at rest, noise-free renderer). During motion the body pitches and rolls, which
# tilts the camera, so the errors are larger. Two corrections are applied:
# - a floor on each standard deviation (an ASSUMPTION covering effects neither characterisation sees, such as pixel
#   noise of a real camera), and
# - a per-component "dynamic_inflation" factor MEASURED by experiments/camera_noise_while_driving.py, which compares measurements taken
#   during docking runs with the measurement the true pose would give, and
# - a further yaw scale for the correlation of the yaw errors over consecutive frames, chosen by the consistency tuning
#   in experiments/tune_kalman_filter.py (settings/kalman_filter_tuning.json).
# ==============================================================================================================

NOISE_JSON = Path(__file__).resolve().parents[1] / "settings" / "camera_noise.json"
TUNING_JSON = Path(__file__).resolve().parents[1] / "settings" / "kalman_filter_tuning.json"

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
        also the yaw scale for correlated errors chosen by experiments/tune_kalman_filter.py (settings/kalman_filter_tuning.json)."""
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

"""ArUco detection, marker pose by solvePnP, and conversion to a MarkerMeasurement in the base frame B.

Chain (docs/frames.md):
1. detectMarkers gives the four image corners of marker id 0 (sub-pixel refined).
2. solvePnP with SOLVEPNP_IPPE_SQUARE gives the marker pose in the OpenCV camera frame. The square marker has a
   two-fold pose ambiguity, so both IPPE solutions are computed and the one with the lower reprojection error is kept.
3. OpenCV camera frame to MuJoCo camera frame: R_MJ_FROM_CV = diag(1, -1, -1).
4. MuJoCo camera frame to base frame B: the camera extrinsics from TB3Sim.sensor_extrinsics().
5. OpenCV marker frame (x right, y up, z out of the face) to the project marker frame M (x out of the face, y right,
   z up): x_M = z_cv, y_M = x_cv, z_M = y_cv.
6. Range-bias correction from the calibration (see to_base_frame).
7. Planar projection: base_T_marker = (x, y, yaw) where yaw is the direction of the marker normal x_M in B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src import known_map as km
from src.tb3.interfaces import MarkerMeasurement
from src.vision.camera_model import DIST_COEFFS, R_MJ_FROM_CV, intrinsics

# Second IPPE solution passed on when its reprojection error is below this multiple of the best one. In the vision
# characterisation, flipped detections had a median ratio of 1.4 and correct ones a median of 30 (docs/vision.md).
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
    range: float               # planar distance from base_footprint to the marker centre, m
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
    the same ray (docs/vision.md, fitted by evaluation/vision_characterisation.py).
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

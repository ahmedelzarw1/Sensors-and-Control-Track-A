"""Pinhole model of the simulated robot camera and the MuJoCo to OpenCV axis conversion.

The resolution and vertical field of view are set explicitly in the generated scene (scripts/make_track_a_scene.py
writes them into models/track_a_room.xml), so the intrinsics follow from them exactly:

    fy = (H / 2) / tan(fovy / 2),   fx = fy (MuJoCo renders square pixels),
    cx = (W - 1) / 2,   cy = (H - 1) / 2   (OpenCV puts pixel centres on integer coordinates)

There is no lens distortion in the renderer, so the distortion coefficients are zero. The intrinsics are checked
indirectly by the vision characterisation (evaluation/vision_characterisation.py): after the range-bias correction
the mean pose error is below 1 mm at short range. A dedicated projection test is still to be written.

Axis conventions (docs/frames.md):
- MuJoCo camera frame: x to the image right, y up, looking along -z.
- OpenCV camera frame: x to the image right, y down, looking along +z.
So R_mujoco_from_opencv = diag(1, -1, -1).
"""
from __future__ import annotations

import numpy as np

# Chosen so that the 0.15 m marker is detected from about 0.2 m (docked) to beyond 2.5 m. See docs/vision.md.
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

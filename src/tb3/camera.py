"""Simulated robot camera: offscreen rendering of the fixed forward camera `color_camera`.

This is the simulator side of the camera (it produces images, like the LiDAR ray casting in platform.py). Image
processing is in src/vision/.

Rendering backend. MuJoCo picks the OpenGL backend from the MUJOCO_GL environment variable. On macOS the default (CGL)
works and was used for every result in this repository. On Linux without a display, set MUJOCO_GL=egl (GPU) or
MUJOCO_GL=osmesa (software, slow) before Python starts. The Linux backends have not been tested by us. See docs/vision.md.
"""
from __future__ import annotations

import mujoco
import numpy as np

from src.vision.camera_model import CAMERA_FOVY_DEG, CAMERA_HEIGHT, CAMERA_WIDTH


class SimCamera:
    def __init__(self, sim, name: str = "color_camera", width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self._sim = sim
        self.name = name
        self.width, self.height = width, height
        cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam < 0:
            raise ValueError(f"camera {name!r} not in model")
        self.fovy_deg = float(sim.model.cam_fovy[cam])
        if abs(self.fovy_deg - CAMERA_FOVY_DEG) > 1e-6:
            raise ValueError(f"scene fovy {self.fovy_deg} differs from camera_model.CAMERA_FOVY_DEG; regenerate the scene")
        self._renderer = mujoco.Renderer(sim.model, height, width)

    def render(self) -> np.ndarray:
        """RGB image (H, W, 3) uint8 with row 0 at the top, as OpenCV expects."""
        self._renderer.update_scene(self._sim.data, camera=self.name)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()

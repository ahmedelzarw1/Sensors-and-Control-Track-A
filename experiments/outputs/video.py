"""Demo video: top-down view (true path, estimated path, 3 sigma ellipse) next to the robot camera view.

Used as the frame_hook of experiments.run_simulation.run_episode. The true pose drawn here comes from the episode log, which the
runner fills from experiments.true_position. It is shown for evaluation only and never reaches the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from experiments.outputs.draw_room import draw_map, ellipse_points  # noqa: E402
from robot_code.room import room_map as km  # noqa: E402

# drawFrameAxes warns when an axis tip falls outside the image (normal when docked close to the marker); the warning
# does not affect the drawing of the visible part, so only errors are logged.
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

W, H = 640, 480


def _tri(pose, size=0.12):
    x, y, th = pose
    pts = np.array([[size, 0.0], [-0.6 * size, 0.5 * size], [-0.6 * size, -0.5 * size], [size, 0.0]])
    c, s = np.cos(th), np.sin(th)
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


class DemoVideo:
    def __init__(self, path: Path, fps: int = 10, title: str = "", layout=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=7, macro_block_size=8)
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0.10, 0.11, 0.88, 0.82])
        draw_map(self.ax, layout=layout)
        self.ax.set_title(title, fontsize=9)
        self.ref_line, = self.ax.plot([], [], color="0.55", lw=1, ls="--", label="planned reference")
        self.true_line, = self.ax.plot([], [], "b", lw=1.5, label="true path")
        self.est_line, = self.ax.plot([], [], "r", lw=1.0, label="EKF estimate")
        self.ell, = self.ax.plot([], [], "r", lw=0.8, label="3 sigma position")
        self.true_tri, = self.ax.plot([], [], "b", lw=1.2)
        self.est_tri, = self.ax.plot([], [], "r", lw=1.2)
        self.text = self.ax.text(km.ROOM_X_MIN + 0.05, km.ROOM_Y_MAX - 0.08, "", fontsize=8, va="top",
                                 bbox=dict(fc="w", ec="0.7", alpha=0.85))
        self.ax.legend(fontsize=6, loc="lower left")
        self._plan_id = None

    def __call__(self, k, image, mission, log):
        x, y = np.array(log["x"]), np.array(log["y"])
        xh, yh = np.array(log["xh"]), np.array(log["yh"])
        self.true_line.set_data(x, y)
        self.est_line.set_data(xh, yh)
        if mission.ref is not None and id(mission.ref) != self._plan_id:
            self.ref_line.set_data(mission.ref.x, mission.ref.y)
            self._plan_id = id(mission.ref)
        P = np.array([[log["Pxx"][-1], log["Pxy"][-1]], [log["Pxy"][-1], log["Pyy"][-1]]])
        e = ellipse_points([xh[-1], yh[-1]], P)
        self.ell.set_data(e[:, 0], e[:, 1])
        tt = _tri((x[-1], y[-1], log["th"][-1]))
        te = _tri((xh[-1], yh[-1], log["thh"][-1]))
        self.true_tri.set_data(tt[:, 0], tt[:, 1])
        self.est_tri.set_data(te[:, 0], te[:, 1])
        err = np.hypot(x[-1] - xh[-1], y[-1] - yh[-1])
        fix = "accepted" if log["accepted"][-1] else ("gated out" if log["detected"][-1] else "not seen")
        self.text.set_text(f"t = {log['t'][-1]:5.1f} s   phase: {log['phase'][-1]}\n"
                           f"marker: {fix}\nestimation error: {1000 * err:5.1f} mm\n"
                           f"3 sigma: {3000 * np.sqrt(log['Pxx'][-1] + log['Pyy'][-1]):5.1f} mm")
        self.canvas.draw()
        left = np.asarray(self.canvas.buffer_rgba())[:, :, :3]
        right = self._camera_panel(image, mission)
        self.writer.append_data(np.hstack([left, right]))

    def _camera_panel(self, image, mission):
        return annotate_camera(image, mission)

    def close(self):
        self.writer.close()
        plt.close(self.fig)


def annotate_camera(image, mission, caption: str = "robot camera (axes: x red, y green, z blue)"):
    """The robot camera image with the detected marker outlined, its pose axes and the measured range drawn on it."""
    img = np.ascontiguousarray(image[:, :, ::-1])  # OpenCV draws in BGR order
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H))
    det = mission.sensor.last_detection
    if det is not None:
        pts = det.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        rvec, _ = cv2.Rodrigues(det.R_cam_marker)
        cv2.drawFrameAxes(img, mission.sensor.detector.K, mission.sensor.detector.dist, rvec,
                          det.t_cam_marker.reshape(3, 1), 0.07, 2)
        pm = mission.sensor.last_planar
        cv2.putText(img, f"id {det.marker_id}  range {pm.range:.2f} m  view {np.rad2deg(pm.view_angle):.0f} deg",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(img, "no marker detected", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
    cv2.putText(img, caption, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.ascontiguousarray(img[:, :, ::-1])

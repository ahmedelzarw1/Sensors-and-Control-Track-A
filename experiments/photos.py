"""Camera taking photos: snapshots of the robot camera and of the room at the key moments of a run.

A photo is saved at the start, when the marker is first detected (the camera is only on at the end), at every phase
change, and at the end. Each photo
is the annotated robot camera image (marker outline, pose axes, measured range) next to a top-down view of the room
with the true path, the estimated path and the 3 sigma ellipse at that moment. contact_sheet.png puts them all on one
page.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.plots import draw_map, ellipse_points, lidar_map_rgba, robot_triangle  # noqa: E402
from robot_code import room_map as km  # noqa: E402
from experiments.video import annotate_camera, reference_in_room  # noqa: E402


class PhotoTaker:
    def __init__(self, out_dir: Path, layout=None, every_s: float = 0.0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layout = layout
        self.every_s = every_s
        self.photos: list[tuple[Path, str]] = []
        self._phase = None
        self._seen = False
        self._last_t = -1e9
        self._last = None

    def _shoot(self, image, mission, log, label: str) -> None:
        t = log["t"][-1]
        cam = annotate_camera(image, mission, caption=f"robot camera  t = {t:.1f} s  {label}")
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        draw_map(ax, layout=self.layout)
        if getattr(mission, "lidar_map", None) is not None:
            img, extent = lidar_map_rgba(mission.lidar_map)
            ax.imshow(img, extent=extent, origin="lower", zorder=1, interpolation="nearest")
            ax.set_xlim(km.ROOM_X_MIN - 0.1, km.ROOM_X_MAX + 0.1)
            ax.set_ylim(km.ROOM_Y_MIN - 0.1, km.ROOM_Y_MAX + 0.1)
        ax.plot(log["x"], log["y"], "b", lw=1.5, label="true path")
        ax.plot(log["xh"], log["yh"], "r", lw=1.0, label="navigation EKF")
        ax.plot(log["xd"], log["yd"], "m", lw=1.0, label="docking EKF")
        if mission.ref is not None:
            ax.plot(*reference_in_room(mission, log), color="0.55", lw=1, ls="--", label="planned reference")
        P = np.array([[log["Pxx"][-1], log["Pxy"][-1]], [log["Pxy"][-1], log["Pyy"][-1]]])
        e = ellipse_points([log["xh"][-1], log["yh"][-1]], P)
        ax.plot(e[:, 0], e[:, 1], "r", lw=0.8)
        robot_triangle(ax, (log["x"][-1], log["y"][-1], log["th"][-1]), "b")
        ax.set_title(f"t = {t:.1f} s: {label}", fontsize=9)
        ax.legend(fontsize=6, loc="lower left")
        fig.tight_layout()
        fig.canvas.draw()
        top = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        img = np.hstack([top, cam])
        name = f"{len(self.photos):02d}_t{t:05.1f}_{label.replace(' ', '_').replace(':', '')}.png"
        imageio.imwrite(self.dir / name, img)
        self.photos.append((self.dir / name, f"t = {t:.1f} s: {label}"))

    def __call__(self, k, image, mission, log):
        self._last = (image, mission, log)
        phase = mission.info["phase"]
        if k == 0:
            self._shoot(image, mission, log, "start")
        if not self._seen and mission.info["meas"] is not None:
            self._seen = True
            self._shoot(image, mission, log, "marker first detected")
        if self._phase is not None and phase != self._phase:
            self._shoot(image, mission, log, f"phase {self._phase} to {phase}")
        elif self.every_s > 0 and log["t"][-1] - self._last_t >= self.every_s:
            self._shoot(image, mission, log, phase)
            self._last_t = log["t"][-1]
        self._phase = phase

    def finish(self, result=None) -> Path:
        """Final photo and the contact sheet. Returns the contact sheet path."""
        last_t = self._last[2]["t"][-1] if self._last is not None else None
        already = bool(self.photos) and self.photos[-1][1].startswith(f"t = {last_t:.1f} s")
        if self._last is not None and not already:
            label = "end" if result is None else f"end ({result.status})"
            self._shoot(*self._last, label)
        imgs = [imageio.imread(p) for p, _ in self.photos]
        cols = 2
        rows = int(np.ceil(len(imgs) / cols))
        h, w = imgs[0].shape[:2]
        sheet = np.full((rows * h, cols * w, 3), 255, np.uint8)
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im[:, :, :3]
        path = self.dir.parent / "contact_sheet.png"
        imageio.imwrite(path, sheet)
        return path

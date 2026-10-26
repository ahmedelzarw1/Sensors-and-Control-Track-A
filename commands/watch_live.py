"""Watch the docking mission live in the interactive MuJoCo viewer.

macOS:  ~/.venvs/track-a/bin/mjpython commands/watch_live.py --random
Linux:  python commands/watch_live.py --random

Options: --random [SEED] (a random room, see robot_code/random_room.py; without SEED a new one is drawn),
         --start <point> (the fixed demo room from a start point, see robot_code/room_map.py), --seed N,
         --speed 2.0 (faster than real time),
         --monitor (print what the robot calculates, serial-monitor style; always on with --random),
         --no-panel (hide the live video panel), --panel-size 0.5 (its width as a fraction of the window),
         --record (also save the video panel to results/live/<run>/demo.mp4).
The live video panel in the corner of the viewer shows the top-down view (true path, navigation EKF estimate with its
3 sigma ellipse, docking EKF estimate, planned path, LiDAR obstacle map) next to the robot camera, the same view as the
demo videos.
Mouse: left drag rotates, right drag pans, scroll zooms. Double-click a body to select it. Close the window to stop.
The pipeline is exactly the one in commands/run_demo.py; the viewer only displays the simulator state.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from experiments.serial_monitor import SerialMonitor, print_layout  # noqa: E402
from experiments.video import DemoVideo  # noqa: E402
from experiments.run_simulation import EpisodeConfig, run_episode  # noqa: E402
from robot_code import room_map as km  # noqa: E402
from robot_code import random_room as rl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=list(km.START_POINTS), default="west_centre")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time")
    ap.add_argument("--random", nargs="?", const=-1, type=int, help="random layout (optional seed)")
    ap.add_argument("--monitor", action="store_true", help="print what the robot calculates")
    ap.add_argument("--no-panel", action="store_true", help="hide the live video panel")
    ap.add_argument("--panel-size", type=float, default=0.5, help="panel width as a fraction of the window width")
    ap.add_argument("--record", action="store_true", help="also save the video panel as an MP4")
    args = ap.parse_args()
    if args.random is not None:
        seed = args.random if args.random >= 0 else int(time.time()) % 100000
        layout, report = rl.sample_layout(seed)
        args.seed, args.monitor = seed, True
        print_layout(layout, report)
    else:
        layout = km.default_layout(args.start)

    state = {"viewer": None, "t0": None, "phase": None, "sim": None, "cam": None, "panel": None, "runs": 0}

    def show_panel(v, mission):
        """Overlay the latest video-panel frame in the bottom-right corner, and live values in the top-left."""
        panel = state["panel"]
        vp = v.viewport
        if panel is not None and panel.last_frame is not None and vp is not None and vp.width > 0:
            w = int(vp.width * args.panel_size)
            h = int(w * panel.last_frame.shape[0] / panel.last_frame.shape[1])
            img = cv2.resize(panel.last_frame, (w, h), interpolation=cv2.INTER_AREA)
            v.set_images((mujoco.MjrRect(vp.width - w - 10, 10, w, h), img))
        info = mission.info
        est = info.get("est")
        if est is not None:
            meas = info.get("meas")
            if info["phase"] in ("look", "servo", "done"):
                marker = (f"{meas.range_bearing[0]:.2f} m, {'used' if info['accepted'] else 'rejected'}"
                          if meas is not None else "not seen")
            else:
                marker = "camera off"
            lidar = ("used" if info["lidar_accepted"] else "rejected") if info["lidar_fix"] is not None else "-"
            sd = 1000 * np.sqrt(max(est.P[0, 0] + est.P[1, 1], 0.0))
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         "time\nphase\nestimate x, y\nheading\nuncertainty (1 sigma)\ncommand v, w\nLiDAR fix\n"
                         "LiDAR front\nmarker",
                         f"{mission.t:.1f} s\n{info['phase']}\n{est.x_hat[0]:.3f}, {est.x_hat[1]:.3f} m\n"
                         f"{np.rad2deg(est.x_hat[2]):.1f} deg\n{sd:.1f} mm\n"
                         f"{info['cmd'].v:.3f} m/s, {info['cmd'].omega:.3f} rad/s\n{lidar}\n"
                         f"{info['front']:.2f} m\n{marker}"))

    def make_panel():
        if args.no_panel:
            return None
        state["runs"] += 1
        name = f"random_{args.seed}" if args.random is not None else f"{args.start}_{args.seed}"
        path = ROOT / "results" / "live" / f"{name}_run{state['runs']}" / "demo.mp4" if args.record else None
        title = f"random room, seed {args.seed}" if args.random is not None else f"demo room from '{args.start}' (seed {args.seed})"
        state["panel"] = DemoVideo(path, title=title, layout=layout)
        return state["panel"]

    def open_viewer(sim):
        v = mujoco.viewer.launch_passive(sim.model, sim.data)
        if state["cam"] is None:
            v.cam.lookat[:] = [0.0, 0.0, 0.0]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 5.5, -55.0, 90.0
        else:  # keep the view the user chose before a reset
            v.cam.lookat[:], v.cam.distance, v.cam.elevation, v.cam.azimuth = state["cam"]
        return v

    def reset_pressed(sim) -> bool:
        """The viewer's Reset (button or Backspace) sets the simulation time back to zero behind the pipeline."""
        return sim.data.time < sim.t - 0.05

    def step_hook(sim, mission):
        if state["viewer"] is None:
            state["viewer"] = open_viewer(sim)
            state["t0"] = time.time() - sim.t / args.speed
        state["sim"] = sim
        v = state["viewer"]
        if not v.is_running():
            raise KeyboardInterrupt
        if mission.phase.value != state["phase"]:
            state["phase"] = mission.phase.value
            if not args.monitor:
                print(f"t = {sim.t:5.1f} s  phase: {state['phase']}")
        show_panel(v, mission)
        v.sync()
        if reset_pressed(sim):
            raise ViewerReset
        lag = state["t0"] + sim.t / args.speed - time.time()
        if lag > 0:
            time.sleep(lag)

    def close_viewer():
        v = state["viewer"]
        if v is not None:
            state["cam"] = (v.cam.lookat.copy(), v.cam.distance, v.cam.elevation, v.cam.azimuth)
            v.close()
        state["viewer"], state["phase"] = None, None

    cfg = EpisodeConfig(seed=args.seed, layout=layout)
    try:
        while True:
            try:
                hooks = [SerialMonitor(every=5)] if args.monitor else []
                panel = make_panel()
                if panel is not None:
                    hooks.append(panel)
                try:
                    res = run_episode(cfg, step_hook=step_hook, frame_hook=hooks)
                finally:
                    if panel is not None:
                        panel.close()
                        if panel.path is not None:
                            print(f"video saved: {panel.path.relative_to(ROOT)}")
                e = res.final_error
                print(f"{res.status}: {1000 * e[0]:.1f} mm along, {1000 * e[1]:.1f} mm lateral, "
                      f"{abs(e[2]) * 57.3:.2f} deg. Press Reset (or Backspace) in the viewer to run again, or close the "
                      "window to exit.")
                v, sim = state["viewer"], state["sim"]
                while v is not None and v.is_running():
                    v.sync()
                    if reset_pressed(sim):
                        raise ViewerReset
                    time.sleep(0.05)
                break
            except ViewerReset:
                print("\nReset pressed: restarting the run from the start point (robot, estimators, planner and "
                      "controller all start again).\n")
                close_viewer()
    except KeyboardInterrupt:
        print("viewer closed")


class ViewerReset(Exception):
    """Raised when the Reset of the MuJoCo viewer is used, so the whole run restarts consistently."""


if __name__ == "__main__":
    main()

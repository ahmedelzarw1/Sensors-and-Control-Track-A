"""Watch the docking mission live in the interactive MuJoCo viewer.

macOS:  ~/.venvs/track-a/bin/mjpython commands/watch_live.py --start centre
Linux:  python commands/watch_live.py --start centre

Options: --start <pad> (see robot_code/room/room_map.py), --seed N, --speed 2.0 (faster than real time),
         --random [SEED] (a random layout, see robot_code/room/random_room.py; without SEED a new one is drawn),
         --monitor (print what the robot calculates, serial-monitor style; always on with --random).
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

import mujoco.viewer  # noqa: E402

from experiments.outputs.serial_monitor import SerialMonitor, print_layout  # noqa: E402
from experiments.run_simulation import EpisodeConfig, run_episode  # noqa: E402
from robot_code.room import room_map as km  # noqa: E402
from robot_code.room import random_room as rl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=list(km.ALL_START_POSES), default="centre")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time")
    ap.add_argument("--random", nargs="?", const=-1, type=int, help="random layout (optional seed)")
    ap.add_argument("--monitor", action="store_true", help="print what the robot calculates")
    args = ap.parse_args()
    layout = None
    if args.random is not None:
        seed = args.random if args.random >= 0 else int(time.time()) % 100000
        layout, report = rl.sample_layout(seed)
        args.seed, args.monitor = seed, True
        print_layout(layout, report)

    state = {"viewer": None, "t0": None, "phase": None, "sim": None, "cam": None}

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

    cfg = EpisodeConfig(start=args.start if layout is None else "random", seed=args.seed, layout=layout)
    try:
        while True:
            try:
                res = run_episode(cfg, step_hook=step_hook,
                                  frame_hook=SerialMonitor(every=5) if args.monitor else None)
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
                print("\nReset pressed: restarting the run from the start pad (robot, estimator, planner and controller "
                      "all start again).\n")
                close_viewer()
    except KeyboardInterrupt:
        print("viewer closed")


class ViewerReset(Exception):
    """Raised when the Reset of the MuJoCo viewer is used, so the whole run restarts consistently."""


if __name__ == "__main__":
    main()

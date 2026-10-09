"""Watch the docking mission live in the interactive MuJoCo viewer.

macOS:  .venv/bin/mjpython scripts/view_live.py --start centre
Linux:  python scripts/view_live.py --start centre

Options: --start <pad> (see src/known_map.py), --seed N, --speed 2.0 (faster than real time).
Mouse: left drag rotates, right drag pans, scroll zooms. Double-click a body to select it. Close the window to stop.
The pipeline is exactly the one in scripts/run_demo.py; the viewer only displays the simulator state.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco.viewer  # noqa: E402

from evaluation.runner import EpisodeConfig, run_episode  # noqa: E402
from src import known_map as km  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=list(km.ALL_START_POSES), default="centre")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time")
    args = ap.parse_args()

    state = {"viewer": None, "t0": None, "phase": None}

    def step_hook(sim, mission):
        if state["viewer"] is None:
            state["viewer"] = mujoco.viewer.launch_passive(sim.model, sim.data)
            v = state["viewer"]
            v.cam.lookat[:] = [0.0, 0.0, 0.0]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 5.5, -55.0, 90.0
            state["t0"] = time.time() - sim.t / args.speed
        v = state["viewer"]
        if not v.is_running():
            raise KeyboardInterrupt
        if mission.phase.value != state["phase"]:
            state["phase"] = mission.phase.value
            print(f"t = {sim.t:5.1f} s  phase: {state['phase']}")
        v.sync()
        lag = state["t0"] + sim.t / args.speed - time.time()
        if lag > 0:
            time.sleep(lag)

    try:
        res = run_episode(EpisodeConfig(start=args.start, seed=args.seed), step_hook=step_hook)
        e = res.final_error
        print(f"{res.status}: {1000 * e[0]:.1f} mm along, {1000 * e[1]:.1f} mm lateral, "
              f"{abs(e[2]) * 57.3:.2f} deg. Close the viewer window to exit.")
        while state["viewer"] is not None and state["viewer"].is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("viewer closed")


if __name__ == "__main__":
    main()

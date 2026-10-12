"""End-to-end Track A demo: localise from the marker, plan around the obstacles, track with LQR on the EKF estimate,
dock at the ArUco-marked charging station.

    python commands/run_demo.py                     # all five start pads, video of the first one
    python commands/run_demo.py --start south       # one start pad
    python commands/run_demo.py --video-all         # a video for every start pad
    python commands/run_demo.py --hidden            # also the two start pads without a view of the marker

Output in results/demo/<start>/: log.csv, path.png, estimation_errors.png, nees_nis.png, control.png, demo.mp4.
results/demo/summary.csv and drift.png pool all runs. The controller only ever receives the EKF estimate; ground
truth is read by experiments/ for the logs, plots and video only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.outputs.run_report import episode_metrics, plot_drift, plot_episode, write_csv  # noqa: E402
from experiments.run_simulation import EpisodeConfig, run_episode  # noqa: E402
from experiments.outputs.video import DemoVideo  # noqa: E402
from robot_code.room import room_map as km  # noqa: E402

OUT = ROOT / "results" / "demo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", choices=list(km.ALL_START_POSES), help="run one start pad only")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--video-all", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--hidden", action="store_true", help="also run the start pads without a view of the marker")
    args = ap.parse_args()

    starts = [args.start] if args.start else list(km.START_POSES) + (list(km.HIDDEN_START_POSES) if args.hidden else [])
    results, rows = [], []
    for i, start in enumerate(starts):
        t0 = time.time()
        cfg = EpisodeConfig(start=start, seed=args.seed)
        want_video = not args.no_video and (args.video_all or i == 0)
        video = DemoVideo(OUT / start / "demo.mp4", title=f"Track A docking from '{start}' (seed {args.seed})") \
            if want_video else None
        res = run_episode(cfg, frame_hook=video)
        if video is not None:
            video.close()
        out = OUT / start
        out.mkdir(parents=True, exist_ok=True)
        write_csv(res, out / "log.csv")
        plot_episode(res, out)
        (out / "events.json").write_text(json.dumps(res.events, indent=1) + "\n")
        m = episode_metrics(res)
        rows.append(m)
        results.append(res)
        print(f"{start:15s} {res.status:17s} dock error: {1000 * m['dock_err_long_m']:6.1f} mm along, "
              f"{1000 * m['dock_err_lat_m']:6.1f} mm lateral, {m['dock_err_heading_deg']:5.2f} deg | "
              f"time {m['time_s']:5.1f} s | max estimation error {1000 * m['max_est_pos_m']:5.1f} mm | "
              f"mean NEES {m['mean_nees']:.2f} | wall {time.time() - t0:4.0f} s")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    drift = plot_drift(results, OUT / "drift.png")
    (OUT / "drift.json").write_text(json.dumps(drift, indent=1) + "\n")
    n_ok = sum(r["success"] for r in rows)
    print(f"\n{n_ok} of {len(rows)} runs docked within tolerance. Results in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

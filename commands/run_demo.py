"""End-to-end Track A demo in the fixed demo room: from a start point at the west end to the charging station at the
east end. Navigation EKF (odometry + LiDAR), LiDAR obstacle map and safety stop, LQR tracking, then visual servoing into
the station with the camera. The robot is told the station position only to within 0.20 m (here it is 0.15 m off).

    python commands/run_demo.py                      # all three start points, video of the first one
    python commands/run_demo.py --start west_north   # one start point
    python commands/run_demo.py --video-all          # a video for every start point

Output in results/demo/<start>/: log.csv, path.png, estimation_errors.png, nees_nis.png, control.png, demo.mp4.
results/demo/summary.csv pools all runs. The controller only ever receives the EKF estimates; ground truth is read by
experiments/ for the logs, plots and video only.
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

from experiments.plots import episode_metrics, plot_episode, write_csv  # noqa: E402
from experiments.run_simulation import EpisodeConfig, run_episode  # noqa: E402
from experiments.video import DemoVideo  # noqa: E402
from robot_code import room_map as km  # noqa: E402

OUT = ROOT / "results" / "demo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", choices=list(km.START_POINTS), help="run one start point only")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--video-all", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    starts = [args.start] if args.start else list(km.START_POINTS)
    rows = []
    for i, start in enumerate(starts):
        t0 = time.time()
        layout = km.default_layout(start)
        cfg = EpisodeConfig(layout=layout, seed=args.seed)
        want_video = not args.no_video and (args.video_all or i == 0)
        video = DemoVideo(OUT / start / "demo.mp4", title=f"Track A docking from '{start}' (seed {args.seed})",
                          layout=layout) if want_video else None
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
        print(f"{start:12s} {res.status:17s} dock error: {1000 * m['dock_err_long_m']:6.1f} mm along, "
              f"{1000 * m['dock_err_lat_m']:6.1f} mm lateral, {m['dock_err_heading_deg']:5.2f} deg | "
              f"time {m['time_s']:5.1f} s | max estimation error {1000 * m['max_est_pos_m']:5.1f} mm | "
              f"mean NEES {m['mean_nees']:.2f} (navigation), {m['dock_mean_nees']:.2f} (docking) | "
              f"wall {time.time() - t0:4.0f} s")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["success"] for r in rows)
    print(f"\n{n_ok} of {len(rows)} runs docked within tolerance. Results in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

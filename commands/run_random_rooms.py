"""Random rooms: a random start point at the west end, a random charging station on the east wall (the robot is told
its position only to within 0.20 m) and random obstacles in between, checked for clashes, then the full pipeline.

    python commands/run_random_rooms.py                       # one random room (a new seed each time, printed)
    python commands/run_random_rooms.py --seed 7              # reproduce room 7 exactly
    python commands/run_random_rooms.py --seed 0 --count 30   # 30 rooms in a row, with a success rate and its interval

While it runs, the terminal works like a serial monitor: the randomised room and the checks it passed, then a line
twice a second with what the robot is calculating (EKF estimate, uncertainty, commands, LiDAR fix, safety-stop
distance, and at the end the marker measurement and the docking estimate), and the events as they happen.

Output in results/random_rooms/seed_<n>/: layout.png (the layout with the clash rules drawn), layout.json, photos/ (robot
camera and room at the key moments), contact_sheet.png, demo.mp4, monitor.txt (the serial-monitor printout), log.csv
and the plots of commands/run_demo.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.scores import wilson_interval  # noqa: E402
from experiments.serial_monitor import SerialMonitor, print_layout  # noqa: E402
from experiments.photos import PhotoTaker  # noqa: E402
from experiments.plots import draw_map, robot_triangle  # noqa: E402
from experiments.plots import episode_metrics, plot_episode, write_csv  # noqa: E402
from experiments.run_simulation import EpisodeConfig, run_episode  # noqa: E402
from experiments.video import DemoVideo  # noqa: E402
from robot_code import room_map as km  # noqa: E402
from robot_code import random_room as rl  # noqa: E402
from robot_code.path_planning import known_map_grid  # noqa: E402
from robot_code.path_planning import plan_reference  # noqa: E402

OUT = ROOT / "results" / "random_rooms"


class Tee:
    """Write to several streams at once (terminal and monitor.txt)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def plot_layout(layout: km.Layout, report, path: Path) -> None:
    """The room with the clash rules drawn: bounding circles, the 0.40 m gap, the docking corridor, the start point."""
    fig, ax = plt.subplots(figsize=(8, 6.2))
    draw_map(ax, layout=layout)
    for ob in layout.obstacles:
        ax.add_patch(mpatches.Circle((ob.x, ob.y), ob.bounding_radius, fill=False, ls="--", lw=0.8, ec="0.3"))
        ax.add_patch(mpatches.Circle((ob.x, ob.y), ob.bounding_radius + rl.GAP / 2, fill=False, ls=":", lw=0.8,
                                     ec="tab:red"))
        ax.annotate(ob.name.replace("random_", ""), (ob.x, ob.y), fontsize=7, ha="center", va="center")
    x0, x1, y0, y1 = rl.corridor_box(layout)
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="tab:green", alpha=0.12, ec="tab:green", ls="--"))
    ax.add_patch(mpatches.Circle(layout.start[:2], rl.START_OBSTACLE_CLEARANCE, fill=False, ec="tab:blue", ls=":"))
    robot_triangle(ax, layout.start, "tab:blue")
    plan = plan_reference(known_map_grid(layout=layout), layout.start, km.approach_pose(layout.dock_hint_y), 0.1)
    if plan is not None:
        ax.plot(plan.ref.x, plan.ref.y, "k--", lw=1)
    ax.plot([], [], ls="--", c="0.3", label="obstacle bounding circle")
    ax.plot([], [], ls=":", c="tab:red", label="half of the 0.40 m gap rule (red circles never overlap)")
    ax.plot([], [], c="tab:green", alpha=0.5, lw=6, label="docking corridor kept clear")
    ax.plot([], [], c="tab:purple", alpha=0.25, lw=6, label="given station position +- 0.20 m")
    ax.plot([], [], ls=":", c="tab:blue", label="start clearance")
    ax.plot([], [], "k--", label="path on the full map (feasibility check)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"{layout.name}: {len(layout.obstacles)} obstacles, {report.tries} draw(s), "
                 f"smallest gap {report.min_obstacle_gap:.2f} m, station {1000 * report.dock_hint_error:+.0f} mm "
                 "from the given position", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_one(seed: int, args) -> dict:
    layout, report = rl.sample_layout(seed)
    out = OUT / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    # The monitor output always goes to monitor.txt, and to the terminal unless --quiet.
    log_file = open(out / "monitor.txt", "w")
    tee = Tee(log_file) if args.quiet else Tee(sys.stdout, log_file)
    print_layout(layout, report, out=tee)
    (out / "layout.json").write_text(json.dumps({"layout": layout.to_dict(), "checks": asdict(report)}, indent=1) + "\n")
    plot_layout(layout, report, out / "layout.png")

    hooks = [SerialMonitor(every=args.every, out=tee)]
    photos = None if args.no_photos else PhotoTaker(out / "photos", layout=layout)
    if photos:
        hooks.append(photos)
    video = None if args.no_video else DemoVideo(out / "demo.mp4", title=f"random room, seed {seed}", layout=layout)
    if video:
        hooks.append(video)
    res = run_episode(EpisodeConfig(layout=layout, seed=seed), frame_hook=hooks)
    if video:
        video.close()
    sheet = photos.finish(res) if photos else None
    write_csv(res, out / "log.csv")
    plot_episode(res, out, title=f"{layout.name}: {res.status}")
    (out / "events.json").write_text(json.dumps(res.events, indent=1) + "\n")
    m = episode_metrics(res)
    m.update(n_obstacles=len(layout.obstacles), path_length_m=report.path_length,
             min_obstacle_gap_m=report.min_obstacle_gap)
    e = res.final_error
    log_file.write(f"RESULT: {res.status}, dock error {1000 * e[0]:+.1f} mm along, {1000 * e[1]:+.1f} mm lateral, "
                   f"{np.rad2deg(e[2]):+.2f} deg, time {res.duration:.1f} s\n")
    log_file.close()
    print(f"RESULT seed {seed}: {res.status} | dock error {1000 * e[0]:+.1f} mm along, {1000 * e[1]:+.1f} mm lateral, "
          f"{np.rad2deg(e[2]):+.2f} deg | time {res.duration:.1f} s | max estimation error "
          f"{1000 * m['max_est_pos_m']:.1f} mm | output {out.relative_to(ROOT)}/" +
          (f" (contact sheet {sheet.name})" if sheet else ""))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, help="layout seed (default: a new random seed, printed)")
    ap.add_argument("--count", type=int, default=1, help="number of consecutive seeds to run")
    ap.add_argument("--every", type=int, default=5, help="monitor line every N control steps (5 = twice a second)")
    ap.add_argument("--quiet", action="store_true", help="no serial monitor output")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-photos", action="store_true")
    args = ap.parse_args()
    seed0 = args.seed if args.seed is not None else int(time.time()) % 100000
    if args.seed is None:
        print(f"using random seed {seed0} (rerun with --seed {seed0} to reproduce)")
    rows = []
    for seed in range(seed0, seed0 + args.count):
        rows.append(run_one(seed, args))
    if args.count > 1:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"summary_seeds_{seed0}_to_{seed0 + args.count - 1}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n_ok = sum(r["success"] for r in rows)
        lo, hi = wilson_interval(n_ok, len(rows))
        pos = np.array([1000 * r["dock_err_pos_m"] for r in rows if r["success"]])
        print(f"\n{n_ok} of {len(rows)} random rooms docked ({100 * n_ok / len(rows):.0f} percent, 95 percent Wilson "
              f"interval {100 * lo:.0f} to {100 * hi:.0f} percent)." +
              (f" Docking position error over the successes: median {np.median(pos):.1f} mm, max {pos.max():.1f} mm."
               if pos.size else "") + f"\nSummary: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

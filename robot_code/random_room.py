"""Random rooms: a random start point at the west end, a random charging station on the east wall, and random
obstacles in between, with rules that guarantee nothing clashes.

A layout is drawn from a seeded random generator, so every layout can be reproduced from its seed. Candidates are
drawn until one passes every rule (rejection sampling). The rules, and why:

Start point (given to the robot)
- x = START_X (0.35 m from the west wall), y uniform along the west end, heading uniform within +-45 degrees of east.

Charging station (the end point)
- The robot is told dock_hint_y, uniform in DOCK_HINT_RANGE along the east wall.
- The true station is at dock_y = dock_hint_y + an error uniform within +-DOCK_HINT_ERROR (0.20 m). The robot drives to
  the approach point in front of dock_hint_y, and the camera finds the true marker from there (visual servoing).

Obstacles (3 to 5, each a box or a cylinder, 0.30 m tall)
- Box half sizes 0.08 to 0.30 m with any rotation; cylinder radius 0.08 to 0.20 m.
- Each obstacle is checked through the circle that encloses it (its bounding radius), which is conservative.
- Two obstacles are at least GAP = 0.40 m apart. The robot needs 2 x (0.105 m radius + 0.075 m planner margin) = 0.36 m
  to pass between them, so every gap is passable and no obstacle touches another.
- Every obstacle is at least GAP from the walls, for the same reason.
- No obstacle enters the docking corridor: the strip in front of the station from CORRIDOR_BACK before the approach
  point to the east wall, CORRIDOR_HALF_WIDTH either side of both the true and the given station position. The final
  approach and the camera's view of the marker stay clear.
- No obstacle within START_OBSTACLE_CLEARANCE of the start point.

Feasibility
- The planner must find a path from the start point to the approach point on the room's map. If not, redraw.

The simulator places the robot at the start point plus a random placement error (experiments/run_simulation.py), and
then checks that MuJoCo reports no contact between the robot and any obstacle.
All the numbers above are design choices (assumptions), not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_code import room_map as km
from robot_code.path_planning import known_map_grid
from robot_code.path_planning import PlannerParams, plan_reference

N_OBSTACLES = (3, 5)
BOX_HALF = (0.08, 0.30)
CYL_RADIUS = (0.08, 0.20)
OBSTACLE_HEIGHT = 0.30
GAP = 0.40
CORRIDOR_HALF_WIDTH = 0.45
CORRIDOR_BACK = 0.60                 # m before the approach point
START_OBSTACLE_CLEARANCE = km.ROBOT_RADIUS + 0.15
START_WALL_CLEARANCE = km.ROBOT_RADIUS + 0.20
START_HEADING = np.deg2rad(45.0)
DOCK_HINT_RANGE = (-1.0, 1.0)
DOCK_LIMIT = 1.2                     # m, the station centre stays within this distance of the room centre line


@dataclass
class LayoutReport:
    """What the sampler checked, for printing and for the results folder."""
    seed: int
    tries: int
    min_obstacle_gap: float
    min_wall_gap: float
    start_clearance: float
    dock_hint_error: float
    path_length: float
    rejected: dict = field(default_factory=dict)


def corridor_box(layout: km.Layout) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of the docking corridor."""
    x0 = km.approach_pose(layout.dock_hint_y)[0] - CORRIDOR_BACK
    lo, hi = sorted((layout.dock_y, layout.dock_hint_y))
    return x0, km.ROOM_X_MAX, lo - CORRIDOR_HALF_WIDTH, hi + CORRIDOR_HALF_WIDTH


def _corridor_distance(x: float, y: float, box) -> float:
    """Distance from a point to the docking corridor rectangle (0 inside)."""
    x0, x1, y0, y1 = box
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return float(np.hypot(dx, dy))


def _wall_gap(x: float, y: float, r: float) -> float:
    return min(x - r - km.ROOM_X_MIN, km.ROOM_X_MAX - x - r, y - r - km.ROOM_Y_MIN, km.ROOM_Y_MAX - y - r)


def _draw_obstacle(rng, i: int):
    x = rng.uniform(km.ROOM_X_MIN, km.ROOM_X_MAX)
    y = rng.uniform(km.ROOM_Y_MIN, km.ROOM_Y_MAX)
    if rng.random() < 0.5:
        return km.Box(f"random_box_{i}", x, y, rng.uniform(*BOX_HALF), rng.uniform(*BOX_HALF), OBSTACLE_HEIGHT,
                      yaw=float(rng.uniform(0.0, np.pi)))
    return km.Cylinder(f"random_cylinder_{i}", x, y, rng.uniform(*CYL_RADIUS), OBSTACLE_HEIGHT)


def sample_layout(seed: int, max_tries: int = 20000, hint_offset: float | None = None) -> tuple[km.Layout, LayoutReport]:
    """hint_offset: place the true station exactly this far (random side) from the given position instead of within
    +-DOCK_HINT_ERROR. Used to test what happens when the robot is told the position worse than it assumes."""
    rng = np.random.default_rng(seed)
    rejected = {"obstacle_overlap_or_gap": 0, "obstacle_near_wall": 0, "obstacle_in_dock_corridor": 0,
                "obstacle_at_start": 0, "no_path": 0}
    # The number of obstacles is drawn once, so the rejections below do not bias the layouts towards fewer obstacles.
    n = int(rng.integers(N_OBSTACLES[0], N_OBSTACLES[1] + 1))
    for tries in range(1, max_tries + 1):
        # Start point at the west end, and the charging station (given and true position) on the east wall.
        sy = rng.uniform(km.ROOM_Y_MIN + START_WALL_CLEARANCE, km.ROOM_Y_MAX - START_WALL_CLEARANCE)
        start = np.array([km.START_X, sy, rng.uniform(-START_HEADING, START_HEADING)])
        hint = float(rng.uniform(*DOCK_HINT_RANGE))
        dock = hint + float(rng.uniform(-km.DOCK_HINT_ERROR, km.DOCK_HINT_ERROR))
        if hint_offset is not None:
            side = 1.0 if rng.random() < 0.5 else -1.0
            if abs(hint + side * hint_offset) > DOCK_LIMIT:
                side = -side                              # keep the station inside the room
            dock = hint + side * hint_offset
        shell = km.Layout((), start, dock_y=dock, dock_hint_y=hint)
        box = corridor_box(shell)
        # Place the obstacles one at a time; each gets up to 200 draws before the whole arrangement starts again.
        obs, ok = [], True
        for i in range(n):
            for _ in range(200):
                ob = _draw_obstacle(rng, i)
                r = ob.bounding_radius
                if _wall_gap(ob.x, ob.y, r) < GAP:
                    rejected["obstacle_near_wall"] += 1
                elif _corridor_distance(ob.x, ob.y, box) < r + 0.05:
                    rejected["obstacle_in_dock_corridor"] += 1
                elif np.hypot(ob.x - start[0], ob.y - start[1]) - r < START_OBSTACLE_CLEARANCE:
                    rejected["obstacle_at_start"] += 1
                elif any(np.hypot(ob.x - o.x, ob.y - o.y) - r - o.bounding_radius < GAP for o in obs):
                    rejected["obstacle_overlap_or_gap"] += 1
                else:
                    obs.append(ob)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        layout = km.Layout(tuple(obs), start, dock_y=dock, dock_hint_y=hint, name=f"random/{seed}", seed=seed)
        plan = plan_reference(known_map_grid(layout=layout), start, km.approach_pose(hint), 0.1, PlannerParams())
        if plan is None:
            rejected["no_path"] += 1
            continue
        gaps = [np.hypot(a.x - b.x, a.y - b.y) - a.bounding_radius - b.bounding_radius
                for i, a in enumerate(obs) for b in obs[i + 1:]]
        clear = min(np.hypot(start[0] - o.x, start[1] - o.y) - o.bounding_radius for o in obs)
        report = LayoutReport(seed, tries, float(min(gaps)) if gaps else float("inf"),
                              float(min(_wall_gap(o.x, o.y, o.bounding_radius) for o in obs)), float(clear),
                              dock - hint, float(plan.path.length), rejected)
        return layout, report
    raise RuntimeError(f"no valid layout for seed {seed} in {max_tries} tries: {rejected}")

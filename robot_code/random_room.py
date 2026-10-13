"""Random room layouts: random obstacles and a random start pad, with rules that guarantee nothing clashes.

A layout is drawn from a seeded random generator, so every layout can be reproduced from its seed. Candidates are
drawn until one passes every rule (rejection sampling). The rules, and why:

Obstacles (3 to 5, each a box or a cylinder, 0.30 m tall like the fixed room)
- Box half sizes 0.08 to 0.30 m with any rotation; cylinder radius 0.08 to 0.20 m.
- Each obstacle is checked through the circle that encloses it (its bounding radius), which is conservative.
- Two obstacles are at least GAP = 0.40 m apart. The robot needs 2 x (0.105 m radius + 0.075 m planner margin) = 0.36 m
  to pass between them, so every gap is passable and no obstacle touches another.
- Every obstacle is at least GAP from the walls, for the same reason.
- No obstacle enters the docking corridor: the strip in front of the charging station from 0.6 m before the pre-dock
  pose to the wall, 0.45 m either side of the marker line. The final approach and the view of the marker stay clear.

Start pad
- At least ROBOT_RADIUS + 0.15 m from every obstacle and ROBOT_RADIUS + 0.20 m from every wall, outside the docking
  corridor, and at least 1.0 m from the pre-dock pose (so the robot has to travel). Heading uniform in [-pi, pi).
- By default the marker must be in line of sight within 3.3 m, so the robot can fix its pose before it drives
  (allow_hidden=True drops this rule; hidden starts are much harder, see the README).

Feasibility
- The planner must find a path from the start pad to the pre-dock pose on the layout's map. If not, redraw.

The simulator places the robot at the start pad plus a random placement error (experiments/run_simulation.py). After the MuJoCo
scene is built, experiments/run_simulation.py also checks that MuJoCo reports no contact between the robot and any obstacle.
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
CORRIDOR_BACK = 0.60            # m before the pre-dock pose
START_OBSTACLE_CLEARANCE = km.ROBOT_RADIUS + 0.15
START_WALL_CLEARANCE = km.ROBOT_RADIUS + 0.20
START_MIN_TRAVEL = 1.0
MARKER_VIEW_RANGE = 3.3


@dataclass
class LayoutReport:
    """What the sampler checked, for printing and for the results folder."""
    seed: int
    tries: int
    min_obstacle_gap: float
    min_wall_gap: float
    start_clearance: float
    start_wall_clearance: float
    start_marker_distance: float
    marker_visible_from_start: bool
    path_length: float
    rejected: dict = field(default_factory=dict)


def _corridor_distance(x: float, y: float) -> float:
    """Distance from a point to the docking corridor rectangle (0 inside)."""
    x0 = km.PRE_DOCK_POSE[0] - CORRIDOR_BACK
    dx = max(x0 - x, 0.0, x - km.ROOM_X_MAX)
    dy = max(abs(y - km.DOCK_Y) - CORRIDOR_HALF_WIDTH, 0.0)
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


def sample_layout(seed: int, allow_hidden: bool = False, max_tries: int = 20000) -> tuple[km.Layout, LayoutReport]:
    rng = np.random.default_rng(seed)
    rejected = {"obstacle_overlap_or_gap": 0, "obstacle_near_wall": 0, "obstacle_in_dock_corridor": 0,
                "start_clash": 0, "start_no_marker_view": 0, "no_path": 0}
    # The number of obstacles is drawn once, so the rejections below do not bias the layouts towards fewer obstacles.
    n = int(rng.integers(N_OBSTACLES[0], N_OBSTACLES[1] + 1))
    for tries in range(1, max_tries + 1):
        # Place the obstacles one at a time; each gets up to 200 draws before the whole arrangement starts again.
        obs, ok = [], True
        for i in range(n):
            for _ in range(200):
                ob = _draw_obstacle(rng, i)
                r = ob.bounding_radius
                if _wall_gap(ob.x, ob.y, r) < GAP:
                    rejected["obstacle_near_wall"] += 1
                elif _corridor_distance(ob.x, ob.y) < r + 0.05:
                    rejected["obstacle_in_dock_corridor"] += 1
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
        # Start pad.
        sx = rng.uniform(km.ROOM_X_MIN + START_WALL_CLEARANCE, km.ROOM_X_MAX - START_WALL_CLEARANCE)
        sy = rng.uniform(km.ROOM_Y_MIN + START_WALL_CLEARANCE, km.ROOM_Y_MAX - START_WALL_CLEARANCE)
        start = np.array([sx, sy, rng.uniform(-np.pi, np.pi)])
        clear = min(np.hypot(sx - o.x, sy - o.y) - o.bounding_radius for o in obs)
        if (clear < START_OBSTACLE_CLEARANCE or _corridor_distance(sx, sy) < km.ROBOT_RADIUS
                or np.hypot(sx - km.PRE_DOCK_POSE[0], sy - km.PRE_DOCK_POSE[1]) < START_MIN_TRAVEL):
            rejected["start_clash"] += 1
            continue
        layout = km.Layout(tuple(obs), start, name=f"random/{seed}", seed=seed)
        d_marker = float(np.hypot(sx - km.MARKER_POSE_W[0], sy - km.MARKER_POSE_W[1]))
        visible = km.marker_line_of_sight(start[:2], layout) and d_marker <= MARKER_VIEW_RANGE
        if not allow_hidden and not visible:
            rejected["start_no_marker_view"] += 1
            continue
        plan = plan_reference(known_map_grid(layout=layout), start, km.PRE_DOCK_POSE, 0.1, PlannerParams())
        if plan is None:
            rejected["no_path"] += 1
            continue
        gaps = [np.hypot(a.x - b.x, a.y - b.y) - a.bounding_radius - b.bounding_radius
                for i, a in enumerate(obs) for b in obs[i + 1:]]
        report = LayoutReport(seed, tries, float(min(gaps)) if gaps else float("inf"),
                              float(min(_wall_gap(o.x, o.y, o.bounding_radius) for o in obs)), float(clear),
                              float(_wall_gap(sx, sy, 0.0)), d_marker, bool(visible), float(plan.path.length),
                              rejected)
        return layout, report
    raise RuntimeError(f"no valid layout for seed {seed} in {max_tries} tries: {rejected}")

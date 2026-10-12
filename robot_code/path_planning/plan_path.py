"""Obstacle-aware path from the estimated pose to a goal pose, as a ReferenceTrajectory for the LQR tracker.

plan_reference(grid, start_pose, goal_pose) runs A* to an approach point behind the goal (so the path arrives with
the goal heading), appends the goal, shortcuts, fillets the corners and time-samples with the speed profile.
The grid may come from the known map (Stage 1) or from the LiDAR occupancy map built with the estimated pose
(Stage 2). The planner never sees ground truth: its inputs are the estimate and a grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from robot_code.room import room_map as km
from robot_code.path_planning.astar_grid import Grid, astar, segment_clear
from robot_code.path_planning.smooth_path import DensePath, ProfileLimits, fillet_path, reference_from_path, shortcut, speed_profile
from robot_code.data_types import ReferenceTrajectory


@dataclass(frozen=True)
class PlannerParams:
    # Clearance from obstacle cells required for the robot centre: robot radius plus a safety margin (assumption)
    # covering tracking error and estimation error.
    margin: float = 0.075
    arc_margin: float = 0.045        # smaller margin accepted on the rounded corners
    approach: float = 0.40           # m, straight run into the goal along the goal heading
    departure: float = 0.25          # m, straight run out of the start along the current heading
    departure_max_turn: float = np.deg2rad(100.0)  # larger first turns are made in place instead
    r_max: float = 0.35              # m, largest corner radius
    limits: ProfileLimits = field(default_factory=ProfileLimits)

    @property
    def clearance(self) -> float:
        return km.ROBOT_RADIUS + self.margin


@dataclass
class Plan:
    waypoints: list
    path: DensePath
    ref: ReferenceTrajectory


def plan_reference(grid: Grid, start_pose, goal_pose, Ts: float, params: PlannerParams = PlannerParams(),
                   v_start: float = 0.0, v_end: float = 0.05) -> Optional[Plan]:
    gx, gy, gth = goal_pose
    app = np.array([gx - params.approach * np.cos(gth), gy - params.approach * np.sin(gth)])
    pts = astar(grid, start_pose[:2], app, params.clearance)
    if pts is None:
        return None
    clr = grid.clearance()
    if not segment_clear(clr, grid, app, np.array([gx, gy]), km.ROBOT_RADIUS + params.arc_margin):
        return None
    wps = shortcut(pts, grid, params.clearance)
    # Departure: if the first leg is within departure_max_turn of the current heading, start with a short straight
    # run along the heading so the first turn is a driven arc. Turning in place makes the largest odometry heading
    # errors on this platform (turn in place 0.98 of the odometry value against 0.99 on arcs, README.md, Robot model), so
    # it is kept for large first turns only.
    p0 = np.asarray(start_pose[:2], float)
    h0 = np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
    d0 = wps[1] - wps[0]
    turn = abs(float(np.arctan2(h0[0] * d0[1] - h0[1] * d0[0], h0 @ d0)))
    dep = p0 + params.departure * h0
    if turn <= params.departure_max_turn and segment_clear(clr, grid, p0, dep, km.ROBOT_RADIUS + params.arc_margin):
        pts2 = astar(grid, dep, app, params.clearance)
        if pts2 is not None:
            wps = [p0] + shortcut(pts2, grid, params.clearance)
    wps = _drop_short(wps + [np.array([gx, gy])])
    path = fillet_path(wps, grid, km.ROBOT_RADIUS + params.arc_margin, r_max=params.r_max)
    v = speed_profile(path, params.limits, v_start, v_end)
    ref = reference_from_path(path, v, Ts)
    return Plan(wps, path, ref)


def _drop_short(wps, min_len: float = 0.02):
    out = [np.asarray(wps[0], float)]
    for p in wps[1:]:
        p = np.asarray(p, float)
        if np.linalg.norm(p - out[-1]) > min_len:
            out.append(p)
        else:
            out[-1] = p  # keep the later point (the goal must stay exact)
    return out

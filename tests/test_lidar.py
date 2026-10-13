"""LiDAR map, safety stop and replanning (robot_code/lidar.py)."""
import numpy as np
import pytest

from experiments.run_simulation import EpisodeConfig, run_episode
from robot_code import random_room as rl
from robot_code import room_map as km
from robot_code.data_types import LidarScan, StateEstimate
from robot_code.lidar import LidarMap, front_blocked, front_distance

ANGLES = np.deg2rad(np.arange(-180, 180, 1.0))


def _scan_with_box(robot_xy, box_center, half=0.1):
    """Synthetic scan from a robot at robot_xy (heading 0, LiDAR at the robot centre) seeing one square box."""
    ranges = np.full(len(ANGLES), np.nan)
    for k, a in enumerate(ANGLES):
        d = np.array([np.cos(a), np.sin(a)])
        for r in np.arange(0.12, 3.5, 0.005):
            p = np.asarray(robot_xy) + r * d
            if np.all(np.abs(p - box_center) <= half):
                ranges[k] = r
                break
    return LidarScan(t=0.0, angles=ANGLES, ranges=ranges)


def _est(x, y, th=0.0):
    return StateEstimate(t=0.0, x_hat=np.array([x, y, th]), P=np.eye(3) * 1e-4)


def test_a_seen_box_becomes_an_obstacle_and_free_space_stays_free():
    m = LidarMap(lidar_offset_xy=(0.0, 0.0))
    m.add_scan(_scan_with_box((-1.0, 0.0), np.array([0.0, 0.0])), _est(-1.0, 0.0))
    g = m.grid()
    i, j = g.to_cell(-0.1, 0.0)           # the box face the robot sees
    assert m.obstacles()[i, j]
    i, j = g.to_cell(-0.5, 0.0)           # between the robot and the box
    assert not m.obstacles()[i, j] and m.logodds[i, j] < 0


def test_the_map_is_placed_with_the_estimate():
    # Same scan, estimate shifted by 0.3 m in y: the obstacle is drawn 0.3 m higher. The map only uses the estimate.
    m = LidarMap(lidar_offset_xy=(0.0, 0.0))
    m.add_scan(_scan_with_box((-1.0, 0.0), np.array([0.0, 0.0])), _est(-1.0, 0.3))
    g = m.grid()
    assert m.obstacles()[g.to_cell(-0.1, 0.3)] and not m.obstacles()[g.to_cell(-0.1, 0.0)]


def test_hits_next_to_a_known_wall_are_not_new_obstacles():
    m = LidarMap(lidar_offset_xy=(0.0, 0.0))
    scan = LidarScan(t=0.0, angles=np.array([0.0]), ranges=np.array([km.ROOM_X_MAX - 0.10 - 1.0]))  # 0.10 m from wall
    m.add_scan(scan, _est(1.0, -1.0))
    assert m.obstacles().sum() == 0


def test_safety_stop_and_path_check():
    near = _scan_with_box((0.0, 0.0), np.array([0.25, 0.0]), half=0.05)   # face at 0.20 m ahead
    far = _scan_with_box((0.0, 0.0), np.array([1.0, 0.0]), half=0.05)
    assert front_distance(near, (0.0, 0.0)) == pytest.approx(0.20, abs=0.01)
    assert front_blocked(near, (0.0, 0.0)) and not front_blocked(far, (0.0, 0.0))
    m = LidarMap(lidar_offset_xy=(0.0, 0.0))
    m.add_scan(far, _est(0.0, 0.0))
    assert m.path_blocked(np.linspace(0, 1.5, 30), np.zeros(30), km.ROBOT_RADIUS)          # straight through the box
    assert not m.path_blocked(np.linspace(0, 1.5, 30), np.full(30, 0.6), km.ROBOT_RADIUS)  # well clear of it


def _seed_whose_blind_path_hits_an_obstacle():
    """The first random room where a path planned with no obstacle knowledge (walls only) runs into an obstacle."""
    from robot_code.lidar import LidarMap
    from robot_code.path_planning import plan_reference
    for seed in range(100):
        layout, _ = rl.sample_layout(seed)
        plan = plan_reference(LidarMap().grid(), layout.start, km.PRE_DOCK_POSE, 0.1)
        hits = np.zeros(len(plan.ref.x), bool)
        for ob in layout.obstacles:
            hits |= np.hypot(plan.ref.x - ob.x, plan.ref.y - ob.y) < ob.bounding_radius + km.ROBOT_RADIUS
        if hits.any():
            return seed, layout
    raise AssertionError("no such room in 100 seeds")


def test_robot_discovers_an_obstacle_after_planning_and_replans_around_it():
    # No localisation spin: the first plan is made on an empty map. In this room that plan runs into an obstacle, so
    # the LiDAR has to find it and the robot has to replan, then still dock.
    seed, layout = _seed_whose_blind_path_hits_an_obstacle()
    r = run_episode(EpisodeConfig(layout=layout, seed=seed, start="random", localise_fixes=0))
    events = [e[1] for e in r.events]
    assert any("LiDAR" in e for e in events), (seed, events)
    assert r.success, (seed, r.status, events)

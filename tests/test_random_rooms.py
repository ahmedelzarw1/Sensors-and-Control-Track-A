"""Random layouts: every rule of robot_code/random_room.py holds, and nothing clashes in the simulated world."""
import numpy as np
import pytest

from experiments.run_simulation import robot_obstacle_contacts
from robot_code import room_map as km
from robot_code import random_room as rl
from robot_code.path_planning import known_map_grid
from robot_code.path_planning import plan_reference
from robot_code.simulator import load_room_sim

SEEDS = range(40)


def _footprint_gap(a, b, n=400):
    """Smallest distance between the exact footprints of two obstacles (sampled on a's outline)."""
    if isinstance(a, km.Cylinder):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        X, Y = a.x + a.radius * np.cos(t), a.y + a.radius * np.sin(t)
    else:
        u = np.linspace(-1, 1, n // 4)
        edges = [(u * a.hx, np.full_like(u, a.hy)), (u * a.hx, np.full_like(u, -a.hy)),
                 (np.full_like(u, a.hx), u * a.hy), (np.full_like(u, -a.hx), u * a.hy)]
        lx, ly = np.concatenate([e[0] for e in edges]), np.concatenate([e[1] for e in edges])
        c, s = np.cos(a.yaw), np.sin(a.yaw)
        X, Y = a.x + c * lx - s * ly, a.y + s * lx + c * ly
    assert not np.any(km.footprint_contains(b, X, Y)), "footprints overlap"
    return np.hypot(X - b.x, Y - b.y).min() - (b.radius if isinstance(b, km.Cylinder) else 0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_layout_rules_hold(seed):
    layout, rep = rl.sample_layout(seed)
    obs = layout.obstacles
    assert rl.N_OBSTACLES[0] <= len(obs) <= rl.N_OBSTACLES[1]
    for i, a in enumerate(obs):
        assert rl._wall_gap(a.x, a.y, a.bounding_radius) >= rl.GAP
        assert rl._corridor_distance(a.x, a.y) >= a.bounding_radius + 0.05
        for b in obs[i + 1:]:
            assert np.hypot(a.x - b.x, a.y - b.y) - a.bounding_radius - b.bounding_radius >= rl.GAP
            _footprint_gap(a, b)  # exact footprints do not overlap either
    sx, sy, _ = layout.start
    assert all(np.hypot(sx - o.x, sy - o.y) - o.bounding_radius >= rl.START_OBSTACLE_CLEARANCE for o in obs)
    assert rl._wall_gap(sx, sy, 0.0) >= rl.START_WALL_CLEARANCE - 1e-9
    assert km.marker_line_of_sight(layout.start[:2], layout)
    assert plan_reference(known_map_grid(layout=layout), layout.start, km.PRE_DOCK_POSE, 0.1) is not None


def test_same_seed_gives_same_layout_and_seeds_differ():
    a, _ = rl.sample_layout(5)
    b, _ = rl.sample_layout(5)
    c, _ = rl.sample_layout(6)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != c.to_dict()


def test_obstacle_counts_are_not_biased_towards_three():
    counts = [len(rl.sample_layout(s)[0].obstacles) for s in range(60)]
    assert all(counts.count(n) >= 10 for n in (3, 4, 5))


@pytest.mark.parametrize("seed", range(10))
def test_simulated_world_matches_the_layout_and_nothing_touches(seed):
    import mujoco
    layout, _ = rl.sample_layout(seed)
    sim = load_room_sim(layout)
    sim.reset(layout.start)
    m = sim.model
    for ob in layout.obstacles:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, ob.name)
        assert g >= 0
        assert np.allclose(sim.data.geom_xpos[g][:2], [ob.x, ob.y], atol=1e-9)
    # MuJoCo reports no contact between the robot and anything but the floor, at rest and after settling.
    assert robot_obstacle_contacts(sim) == []
    from robot_code.data_types import ControlCmd
    for _ in range(5):
        sim.step(ControlCmd(0.0, 0.0))
    assert robot_obstacle_contacts(sim) == []

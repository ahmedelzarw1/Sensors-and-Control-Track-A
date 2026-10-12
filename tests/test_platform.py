import numpy as np
import pytest

from evaluation.ground_truth import true_pose
from src.tb3.interfaces import ControlCmd
from src.tb3.platform import OdomNoise, TB3Sim


@pytest.fixture()
def sim():
    return TB3Sim(lidar_noise=False, odom_noise=OdomNoise(0.0, 0.0), seed=0)


def test_geometry_is_read_from_the_model(sim):
    assert sim.params.wheel_radius == pytest.approx(0.033)
    assert sim.params.wheel_separation == pytest.approx(0.16)
    assert sim.params.max_wheel_speed == pytest.approx(6.67)  # ROBOTIS model: 6.67 rad/s x 0.033 m = 0.22 m/s


def test_forward_command_moves_forward_and_odometry_agrees(sim):
    sim.reset((0.0, -2.0, 0.0))
    for _ in range(20):
        obs = sim.step(ControlCmd(0.1, 0.0))
    p = true_pose(sim)
    assert p[0] == pytest.approx(0.2, abs=0.02) and abs(p[1] + 2.0) < 0.02
    assert obs.odom.v == pytest.approx(0.1, abs=0.01) and abs(obs.odom.omega) < 0.02


def test_positive_omega_turns_counter_clockwise(sim):
    sim.reset((0.0, -2.0, 0.0))
    for _ in range(20):
        obs = sim.step(ControlCmd(0.0, 1.0))
    p = true_pose(sim)
    assert p[2] == pytest.approx(2.0, abs=0.15)
    assert np.hypot(p[0], p[1] + 2.0) < 0.03
    assert obs.odom.omega == pytest.approx(1.0, abs=0.05)


def test_lidar_rate_and_range_to_known_obstacle(sim):
    sim.reset((0.0, 0.0, 0.0))
    scans = []
    for _ in range(4):
        obs = sim.step(ControlCmd(0.0, 0.0))
        if obs.lidar is not None:
            scans.append(obs.lidar)
    assert len(scans) == 2  # 5 Hz at a 10 Hz control rate
    s = scans[-1]
    assert len(s.ranges) == 360
    ok = np.isfinite(s.ranges)
    assert s.ranges[ok].min() >= 0.12 and s.ranges[ok].max() <= 3.5
    # The front obstacle box has its near face at x = 0.92 in the MJCF. The LiDAR origin is the site base_scan.
    expected = 0.92 - sim.data.site_xpos[sim._scan][0]
    i0 = int(np.argmin(np.abs(s.angles)))
    assert s.ranges[i0] == pytest.approx(expected, abs=0.02)


def test_camera_looks_along_robot_forward_axis(sim):
    cam = sim.sensor_extrinsics()["camera"]
    optical_axis = -cam["R"][:, 2]
    assert optical_axis @ np.array([1.0, 0.0, 0.0]) > 0.99


# ---------------------------------------------------------------- patched ROBOTIS model (scripts/make_track_a_scene.py)
def _floor_contacts(sim):
    import mujoco
    m, d = sim.model, sim.data
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if floor in (c.geom1, c.geom2):
            g = c.geom2 if c.geom1 == floor else c.geom1
            out.add((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]), int(m.geom_type[g])))
    return out


def test_robot_rests_on_its_tyres_and_caster_only(sim):
    import mujoco
    sim.reset((0.0, -2.0, 0.0))
    for _ in range(10):
        sim.step(ControlCmd(0.0, 0.0))
    touching = _floor_contacts(sim)
    sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    # Tyres (wheel bodies) and the caster sphere touch the floor; the chassis mesh does not (it used to drag).
    assert {b for b, _ in touching} == {"wheel_left", "wheel_right", "base"}
    assert all(t == sphere for b, t in touching if b == "base")


def _drive(sim, seq):
    p0 = true_pose(sim)
    s = th = 0.0
    for v, w in seq:
        obs = sim.step(ControlCmd(v, w))
        s += obs.odom.v * sim.ctrl_dt
        th += obs.odom.omega * sim.ctrl_dt
    p = true_pose(sim)
    return p, p0, s, th


def test_robot_follows_velocity_commands_without_slip(sim):
    # As shipped by ROBOTIS: 39 percent of the commanded distance and 41 percent of the encoder heading on an arc.
    sim.reset((0.0, -2.0, 0.0))
    p, p0, s, _ = _drive(sim, [(0.1, 0.0)] * 30 + [(0.0, 0.0)] * 10)
    assert np.hypot(*(p[:2] - p0[:2])) == pytest.approx(0.30, rel=0.03)
    assert s == pytest.approx(np.hypot(*(p[:2] - p0[:2])), rel=0.02)
    sim.reset((-3.0, -2.0, 0.0))
    p, p0, _, th = _drive(sim, [(0.15, 0.5)] * 30 + [(0.15, 0.0)] * 10)
    assert (p[2] - p0[2]) / th == pytest.approx(1.0, abs=0.02)


def test_chassis_collides_with_obstacles():
    # The chassis is kept off the floor by collision bitmasks, but must still hit walls and obstacles.
    sim = TB3Sim(lidar_noise=False, odom_noise=OdomNoise(0.0, 0.0), seed=0)
    sim.reset((0.0, 0.0, 0.0))
    for _ in range(80):
        sim.step(ControlCmd(0.15, 0.0))
    # The obstacle face is at x = 0.92 and the chassis front is 0.037 m ahead of the base origin.
    assert true_pose(sim)[0] < 0.92 - 0.03


def test_robot_radius_in_the_known_map_encloses_the_collision_geometry(sim):
    import mujoco
    from src import known_map as km
    m, d = sim.model, sim.data
    sim.reset((0.0, 0.0, 0.0))
    b = sim._foot
    R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
    bodies = {b} | {i for i in range(m.nbody) if m.body_parentid[i] == b}
    radius = 0.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in bodies and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]:
            md = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[md]: m.mesh_vertadr[md] + m.mesh_vertnum[md]]
            loc = (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - p) @ R
            radius = max(radius, float(np.hypot(loc[:, 0], loc[:, 1]).max()))
    assert radius <= km.ROBOT_RADIUS < radius + 0.002

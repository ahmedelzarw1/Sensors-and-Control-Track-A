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
    assert sim.params.max_wheel_speed == pytest.approx(12.0)


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
    # The front obstacle box has its near face at x = 0.92 in the MJCF.
    expected = 0.92 - sim.data.xpos[sim._scan][0]
    i0 = int(np.argmin(np.abs(s.angles)))
    assert s.ranges[i0] == pytest.approx(expected, abs=0.02)


def test_camera_looks_along_robot_forward_axis(sim):
    cam = sim.sensor_extrinsics()["camera"]
    optical_axis = -cam["R"][:, 2]
    assert optical_axis @ np.array([1.0, 0.0, 0.0]) > 0.99

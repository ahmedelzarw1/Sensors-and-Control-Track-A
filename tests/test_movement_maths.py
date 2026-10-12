import numpy as np
import pytest
from scipy.signal import cont2discrete

from robot_code.controller.lqr_controller import LQRTracker, design_gain, tvlqr_gains
from robot_code.movement_maths.unicycle_model import (F_unicycle, H_marker_pose, H_marker_range_bearing, ctrb, error_dynamics,
                                error_model, f_unicycle, linearise, observability_matrix_ltv, rank,
                                tracking_error, zoh)
from robot_code.path_planning.reference_path import integrate_reference
from robot_code.data_types import StateEstimate

Ts = 0.1


def num_jac(fun, x0, eps=1e-6):
    cols = []
    for i in range(len(x0)):
        d = np.zeros(len(x0)); d[i] = eps
        cols.append((fun(x0 + d) - fun(x0 - d)) / (2 * eps))
    return np.array(cols).T


@pytest.mark.parametrize("v_r,w_r", [(0.1, 0.0), (0.15, 0.6), (0.05, -1.0), (0.0, 0.5)])
def test_linearisation_matches_finite_differences(v_r, w_r):
    u_ref = np.array([v_r, w_r])
    A_fd = num_jac(lambda e: error_dynamics(e, u_ref, u_ref), np.zeros(3))
    B_fd = num_jac(lambda du: error_dynamics(np.zeros(3), u_ref + du, u_ref), np.zeros(2))
    A, B = linearise(v_r, w_r)
    assert np.allclose(A, A_fd, atol=1e-6) and np.allclose(B, B_fd, atol=1e-6)


def test_zero_error_is_an_equilibrium_on_the_reference():
    u_ref = np.array([0.12, 0.4])
    assert np.allclose(error_dynamics(np.zeros(3), u_ref, u_ref), 0.0)


def test_zoh_matches_scipy():
    A, B = linearise(0.15, 0.6)
    Ad, Bd = zoh(A, B, Ts)
    Ad2, Bd2, *_ = cont2discrete((A, B, np.eye(3), np.zeros((3, 2))), Ts, method="zoh")
    assert np.allclose(Ad, Ad2) and np.allclose(Bd, Bd2)


def test_F_unicycle_matches_finite_differences():
    rng = np.random.default_rng(1)
    for w in (0.0, 0.7, -1.3):
        x = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3, 3)])
        u = np.array([0.15, w])
        fd = num_jac(lambda s: f_unicycle(s, u, Ts), x)
        assert np.allclose(F_unicycle(x, u, Ts), fd, atol=1e-5)


def test_tracking_error_zero_at_reference_and_rotates_into_robot_frame():
    x = np.array([1.0, 2.0, np.pi / 2])
    assert np.allclose(tracking_error(x, x), 0.0)
    # Reference 1 m ahead of a robot facing +y is e_x = 1 (forward), e_y = 0.
    assert np.allclose(tracking_error(x, np.array([1.0, 3.0, np.pi / 2])), [1.0, 0.0, 0.0], atol=1e-12)
    # Reference to the robot's left (towards -x when facing +y) is e_y > 0.
    assert tracking_error(x, np.array([0.0, 2.0, np.pi / 2]))[1] == pytest.approx(1.0)


def test_controllability_rank_depends_on_reference_velocity():
    # Moving reference: controllable. Standstill reference: lateral error cannot be influenced.
    assert rank(ctrb(*error_model(0.1, 0.0, Ts))) == 3
    assert rank(ctrb(*error_model(0.1, 0.5, Ts))) == 3
    assert rank(ctrb(*linearise(0.0, 0.0))) == 2
    assert rank(ctrb(*error_model(0.0, 0.0, Ts))) == 2


def test_design_gain_stabilises_over_operating_grid():
    for v_r in (0.0, 0.02, 0.05, 0.1, 0.22):
        for w_r in (-1.0, 0.0, 1.0, 2.0):
            K = design_gain(v_r, w_r, Ts)
            Ad, Bd = error_model(max(v_r, 0.02), w_r, Ts)
            assert np.all(np.abs(np.linalg.eigvals(Ad - Bd @ K)) < 1.0), (v_r, w_r)


def _reference(n=300):
    v = np.full(n, 0.12)
    w = np.where(np.arange(n) < 100, 0.0, np.where(np.arange(n) < 200, 0.35, -0.2))
    return integrate_reference([0.0, 0.0, 0.0], v, w, Ts)


@pytest.mark.parametrize("mode", ["fixed", "scheduled", "tv"])
def test_closed_loop_converges_from_offset_in_pure_python(mode):
    ref = _reference()
    tr = LQRTracker(Ts, mode=mode, v_nom=0.12)
    x = ref.pose(0) + np.array([0.15, -0.10, 0.25])
    errs = []
    for k in range(len(ref) - 1):
        cmd = tr.command(StateEstimate(t=k * Ts, x_hat=x, P=np.eye(3)), ref, k)
        x = f_unicycle(x, (cmd.v, cmd.omega), Ts)
        errs.append(np.hypot(*tracking_error(x, ref.pose(k + 1))[:2]))
    assert errs[0] < 0.2 and errs[-1] < 0.01
    assert max(errs[100:]) < 0.05


def test_tv_gains_are_finite_and_shaped():
    ref = _reference(100)
    K = tvlqr_gains(ref, Ts)
    assert K.shape == (99, 2, 3) and np.all(np.isfinite(K))


def _window(x0, u, n):
    """Poses and EKF-prediction Jacobians along n steps of constant (v, omega)."""
    xs, Fs = [np.array(x0, dtype=float)], []
    for _ in range(n - 1):
        Fs.append(F_unicycle(xs[-1], u, Ts))
        xs.append(f_unicycle(xs[-1], u, Ts))
    return xs, Fs


def test_single_marker_range_bearing_leaves_rotation_about_the_marker_unobservable():
    m = np.array([2.0, 1.0])
    xs, Fs = _window([0.0, 0.0, 0.3], np.array([0.15, 0.3]), 40)
    O = observability_matrix_ltv(Fs, [H_marker_range_bearing(x, m) for x in xs])
    # Rank stays 2 however long the robot moves: rotating the whole trajectory about the marker
    # leaves range and relative bearing unchanged.
    assert rank(O, tol=1e-7) == 2
    x0 = xs[0]
    n = np.array([m[1] - x0[1], x0[0] - m[0], 1.0])  # rotation about the marker, unit angle
    assert np.allclose(O @ n, 0.0, atol=1e-9)


def test_observability_is_restored_by_marker_pose_or_a_second_marker():
    xs, Fs = _window([0.0, 0.0, 0.3], np.array([0.15, 0.3]), 40)
    # Full marker pose (position and yaw in the base frame): observable from a single view.
    assert rank(observability_matrix_ltv(Fs[:1], [H_marker_pose(xs[0], np.array([2.0, 1.0, 0.0]))])) == 3
    # Two distinct markers with range and bearing only: observable.
    m1, m2 = np.array([2.0, 1.0]), np.array([1.0, -2.0])
    H = [np.vstack([H_marker_range_bearing(x, m1), H_marker_range_bearing(x, m2)]) for x in xs]
    assert rank(observability_matrix_ltv(Fs, H), tol=1e-7) == 3


def test_marker_out_of_view_gives_no_absolute_information():
    xs, Fs = _window([0.0, 0.0, 0.3], np.array([0.15, 0.3]), 10)
    assert rank(observability_matrix_ltv(Fs, [np.zeros((2, 3)) for _ in xs])) == 0

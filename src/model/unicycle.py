"""Differential-drive (unicycle) model, tracking-error model and structural analysis.

State x = [x, y, theta] (pose of base_footprint in W). Input u = [v, omega].
Full derivation and assumptions: docs/model_derivation.md.

Tracking error, expressed in the robot frame (Kanayama form):
    e = [e_x, e_y, e_theta],  [e_x, e_y]^T = R(theta)^T ([x_r, y_r]^T - [x, y]^T),  e_theta = wrap(theta_r - theta)
Error dynamics with applied (v, omega) and reference (v_r, omega_r):
    de_x/dt     =  omega * e_y - v + v_r * cos(e_theta)
    de_y/dt     = -omega * e_x     + v_r * sin(e_theta)
    de_theta/dt =  omega_r - omega
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from src.tb3.interfaces import wrap


# ------------------------------------------------------------------ nonlinear model
def f_unicycle(x: np.ndarray, u: np.ndarray, Ts: float) -> np.ndarray:
    """Exact integration of the unicycle over one sample with (v, omega) held constant."""
    px, py, th = x
    v, w = float(u[0]), float(u[1])
    if abs(w) < 1e-9:
        return np.array([px + v * Ts * np.cos(th), py + v * Ts * np.sin(th), float(wrap(th))])
    thn = th + w * Ts
    return np.array([px + v / w * (np.sin(thn) - np.sin(th)),
                     py - v / w * (np.cos(thn) - np.cos(th)),
                     float(wrap(thn))])


def F_unicycle(x: np.ndarray, u: np.ndarray, Ts: float) -> np.ndarray:
    """Jacobian of f_unicycle with respect to the state (3x3). Used by the EKF prediction step."""
    th = x[2]
    v, w = float(u[0]), float(u[1])
    F = np.eye(3)
    if abs(w) < 1e-9:
        F[0, 2], F[1, 2] = -v * Ts * np.sin(th), v * Ts * np.cos(th)
    else:
        thn = th + w * Ts
        F[0, 2] = v / w * (np.cos(thn) - np.cos(th))
        F[1, 2] = v / w * (np.sin(thn) - np.sin(th))
    return F


def G_unicycle(x: np.ndarray, u: np.ndarray, Ts: float) -> np.ndarray:
    """Jacobian of f_unicycle with respect to the input u = (v, omega) (3x2).

    The EKF uses it to map odometry noise (on the measured v and omega) into pose noise: Q = G diag(s_v^2, s_w^2) G^T.
    """
    th = x[2]
    v, w = float(u[0]), float(u[1])
    G = np.zeros((3, 2))
    G[2, 1] = Ts
    if abs(w) < 1e-6:
        c, s = np.cos(th), np.sin(th)
        G[0, 0], G[1, 0] = Ts * c, Ts * s
        G[0, 1], G[1, 1] = -0.5 * v * Ts ** 2 * s, 0.5 * v * Ts ** 2 * c
        return G
    thn = th + w * Ts
    ds, dc = np.sin(thn) - np.sin(th), np.cos(thn) - np.cos(th)
    G[0, 0], G[1, 0] = ds / w, -dc / w
    G[0, 1] = -v / w ** 2 * ds + v / w * Ts * np.cos(thn)
    G[1, 1] = v / w ** 2 * dc + v / w * Ts * np.sin(thn)
    return G


def tracking_error(x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """Reference minus actual pose, rotated into the robot frame. Returns [e_x, e_y, e_theta]."""
    c, s = np.cos(x[2]), np.sin(x[2])
    dx, dy = x_ref[0] - x[0], x_ref[1] - x[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(x_ref[2] - x[2]))])


def error_dynamics(e: np.ndarray, u: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Continuous-time nonlinear tracking-error dynamics (see module docstring)."""
    ex, ey, eth = e
    v, w = u
    vr, wr = u_ref
    return np.array([w * ey - v + vr * np.cos(eth),
                     -w * ex + vr * np.sin(eth),
                     wr - w])


# ------------------------------------------------------------------ linearisation
def linearise(v_r: float, w_r: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous Jacobians of the error dynamics at e = 0, u = u_ref, in terms of du = u - u_ref.

    A = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],  B = [[-1, 0], [0, 0], [0, -1]].
    """
    A = np.array([[0.0, w_r, 0.0],
                  [-w_r, 0.0, v_r],
                  [0.0, 0.0, 0.0]])
    B = np.array([[-1.0, 0.0],
                  [0.0, 0.0],
                  [0.0, -1.0]])
    return A, B


def zoh(A: np.ndarray, B: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation via the matrix exponential of the augmented system."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * Ts)
    return E[:n, :n], E[:n, n:]


def error_model(v_r: float, w_r: float, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete linearised tracking-error model (Ad, Bd) at a reference operating point."""
    return zoh(*linearise(v_r, w_r), Ts)


# ------------------------------------------------------------------ structural analysis
def ctrb(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    blocks, M = [], B
    for _ in range(n):
        blocks.append(M)
        M = A @ M
    return np.hstack(blocks)


def rank(M: np.ndarray, tol: float = 1e-9) -> int:
    return int(np.linalg.matrix_rank(M, tol=tol))


def observability_matrix_ltv(Fs: list[np.ndarray], Hs: list[np.ndarray]) -> np.ndarray:
    """Observability matrix of a linearised time-varying system over a window of samples.

    Fs[k] maps the state at sample k to sample k+1, Hs[k] is the measurement Jacobian at sample k
    (use a zero-row block when a sensor is not available at that sample).
    O = [H_0; H_1 F_0; H_2 F_1 F_0; ...]. Full column rank means the pose is locally observable.
    """
    rows, Phi = [], np.eye(Fs[0].shape[0])
    for k, H in enumerate(Hs):
        rows.append(H @ Phi)
        if k < len(Fs):
            Phi = Fs[k] @ Phi
    return np.vstack(rows)


def H_marker_range_bearing(x: np.ndarray, marker_xy: np.ndarray) -> np.ndarray:
    """Jacobian of [range, bearing] to a marker at a known world position, for observability analysis.

    h(x) = [hypot(dx, dy), atan2(dy, dx) - theta], dx = m_x - x, dy = m_y - y. The vision/EKF owners
    define the real measurement model; this one is the standard example used for the analysis.
    """
    dx, dy = marker_xy[0] - x[0], marker_xy[1] - x[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / r2, -dx / r2, -1.0]])


def H_marker_pose(x: np.ndarray, marker_pose: np.ndarray) -> np.ndarray:
    """Jacobian of the planar marker pose expressed in the base frame, for a marker with known world pose.

    h(x) = [ R(theta)^T (m_xy - p),  m_yaw - theta ]. Always rank 3.
    """
    c, s = np.cos(x[2]), np.sin(x[2])
    dx, dy = marker_pose[0] - x[0], marker_pose[1] - x[1]
    z1, z2 = c * dx + s * dy, -s * dx + c * dy
    return np.array([[-c, -s, z2],
                     [s, -c, -z1],
                     [0.0, 0.0, -1.0]])

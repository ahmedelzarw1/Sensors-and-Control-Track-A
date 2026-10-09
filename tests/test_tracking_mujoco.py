"""Closed loop in MuJoCo with a PERFECT-ESTIMATE STAND-IN.

The stand-in reads the true pose through evaluation.ground_truth. It exists only here, to check the
model and the LQR on the real simulator before the EKF is available. Real runs use the EKF estimate.
"""
import numpy as np
import pytest

from evaluation.ground_truth import true_pose
from evaluation.metrics import tracking_metrics
from src.control.lqr import LQRTracker
from src.planning.reference import integrate_reference
from src.tb3.interfaces import ControlCmd, StateEstimate
from src.tb3.platform import TB3Sim

Ts = 0.1


def _reference():
    # Straight, then a left arc, then a gentle right arc. Stays clear of the scene's obstacles (x < 0.5).
    n = 220
    k = np.arange(n)
    w = np.where(k < 60, 0.0, np.where(k < 140, 0.30, -0.20))
    return integrate_reference([-3.0, -1.0, 0.0], np.full(n, 0.12), w, Ts)


def _run(mode, offset=(0.10, -0.08, 0.12)):
    ref = _reference()
    sim = TB3Sim(ctrl_dt=Ts, seed=3)
    sim.reset(ref.pose(0) + np.array(offset))
    tr = LQRTracker(Ts, mode=mode, v_nom=0.12)
    poses, cmds = [], []
    for k in range(len(ref)):
        poses.append(true_pose(sim))
        est = StateEstimate(t=sim.t, x_hat=true_pose(sim), P=np.eye(3))  # stand-in, see module docstring
        cmd = tr.command(est, ref, k)
        cmds.append([cmd.v, cmd.omega])
        sim.step(ControlCmd(cmd.v, cmd.omega))
    return tracking_metrics(np.array(poses), ref, np.array(cmds)), ref, np.array(poses)


@pytest.mark.parametrize("mode", ["fixed", "scheduled", "tv"])
def test_lqr_tracks_reference_in_mujoco(mode):
    m, ref, poses = _run(mode)
    print(mode, {k: round(v, 4) for k, v in m.items()})
    tail = poses[100:]
    assert m["final_pos"] < 0.05
    assert m["rms_cross"] < 0.05
    assert m["max_pos"] < 0.2

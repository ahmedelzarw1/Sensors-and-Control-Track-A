"""Run one docking episode in MuJoCo and log everything, including ground truth for evaluation.

The separation that matters: the mission (src/mission.py) receives only odometry, LiDAR scans and camera images. This
file additionally reads the true pose through evaluation.ground_truth, but only to log it next to the estimate. The
true pose never flows back into the mission.

Condition parameters (start pose, odometry noise, wheel scale mismatch, floor friction, velocity impulses, marker
dropout) exist so evaluation/run_trials.py can vary them. Each one changes the simulated world or the sensor stream,
never the pipeline's knowledge of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import numpy as np

from evaluation.ground_truth import true_pose
from src import known_map as km
from src.control.lqr import Limits, LQRTracker
from src.control.slew import SlewRateLimiter
from src.docking.docking import DockTolerance, dock_frame_error
from src.estimation.ekf import EKFParams, MarkerEKF
from src.mission import DockingMission, MissionConfig, Phase
from src.tb3.camera import SimCamera
from src.tb3.platform import ROOM_XML, OdomNoise, TB3Sim, wheel_collision_geom
from src.vision.aruco import MarkerSensor
from src.vision.noise_model import MarkerNoiseModel


@dataclass
class EpisodeConfig:
    start: str = "north_west"
    start_offset: Optional[np.ndarray] = None   # true start = nominal + offset. None: sampled from the prior
    seed: int = 0
    odom_noise: OdomNoise = field(default_factory=OdomNoise)
    ekf: Optional[EKFParams] = None             # None: calibrated and tuned parameters from config/
    tracker_mode: str = "scheduled"
    # Operating limit on the commanded turn rate. The datasheet value 2.84 rad/s is not usable on this platform:
    # with fast turn reversals above about 1.2 rad/s the wheels slipped and the heading error reached 20 degrees in
    # under a second (docs/platform_notes.md). 1.2 rad/s is our choice from that evidence; Stage 3 sweeps it.
    omega_max: float = 1.2
    slew: bool = True
    planner: str = "known_map"
    wheel_scale: tuple = (1.0, 1.0)             # true wheel radius / radius the software believes (left, right)
    friction_scale: float = 1.0                 # multiplies the sliding friction of the wheels and the floor
    impulses: tuple = ()                        # ((t, dv_forward m/s, dw rad/s), ...) applied to the true body
    marker_dropout: float = 0.0                 # probability that a camera frame is lost
    timeout: float = 240.0
    lidar: bool = True
    vision_static_noise_only: bool = False      # True: ignore the measured in-motion inflation (used to measure it)
    vision_r_scale: tuple = (1.0, 1.0, 1.0)     # extra multiplier on the measurement std (x, y, yaw), for tuning
    use_tuning: bool = True                     # False: ignore config/ekf_tuning.json (used by the tuning itself)


@dataclass
class EpisodeResult:
    cfg: EpisodeConfig
    log: dict
    events: list
    plans: list
    success: bool
    status: str
    final_error: np.ndarray      # dock-frame (longitudinal, lateral, heading) of the TRUE final pose
    final_est_error: np.ndarray  # the same for the estimate at the end
    duration: float


def _apply_conditions(sim: TB3Sim, cfg: EpisodeConfig) -> None:
    m = sim.model
    # Wheel radius mismatch: the true geometry changes, the odometry keeps the nominal radius read at construction.
    # The wheel radius mismatch is applied through TB3Sim(odom_radius_scale=...): the tyres are meshes, so the true
    # radius stays as modelled and the robot's software is given a wrong radius instead. The effect on odometry and
    # wheel commands is the same.
    if cfg.friction_scale != 1.0:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        m.geom_friction[g][0] *= cfg.friction_scale
        for name in ("wheel_left", "wheel_right"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            m.geom_friction[wheel_collision_geom(m, b)][0] *= cfg.friction_scale


def _apply_impulse(sim: TB3Sim, dv: float, dw: float) -> None:
    """Add a velocity change to the free joint of the robot (a push)."""
    th = true_pose(sim)[2]
    dof = int(sim.model.jnt_dofadr[sim.model.body_jntadr[sim._foot]])  # noqa: SLF001
    sim.data.qvel[dof + 0] += dv * np.cos(th)
    sim.data.qvel[dof + 1] += dv * np.sin(th)
    sim.data.qvel[dof + 5] += dw


def run_episode(cfg: EpisodeConfig, frame_hook: Optional[Callable] = None,
                step_hook: Optional[Callable] = None) -> EpisodeResult:
    """frame_hook(k, image, mission, log) is called every control step (used for the video).
    step_hook(sim, mission) is called after every control step (used by the live viewer)."""
    rng = np.random.default_rng(cfg.seed)
    nominal = km.ALL_START_POSES[cfg.start].copy()
    offset = rng.normal(0.0, km.START_PRIOR_SIGMA) if cfg.start_offset is None else np.asarray(cfg.start_offset)
    sim = TB3Sim(ROOM_XML, seed=cfg.seed, odom_noise=cfg.odom_noise, odom_radius_scale=cfg.wheel_scale)
    _apply_conditions(sim, cfg)
    sim.reset(nominal + offset)
    cam = SimCamera(sim)
    Ts = sim.ctrl_dt

    ekf_params = cfg.ekf if cfg.ekf is not None else EKFParams.calibrated(use_tuning=cfg.use_tuning)
    ekf = MarkerEKF(nominal, np.diag(km.START_PRIOR_SIGMA ** 2), ekf_params)
    noise = MarkerNoiseModel.from_json(static_only=cfg.vision_static_noise_only, use_tuning=cfg.use_tuning)
    noise.inflation = noise.inflation * np.asarray(cfg.vision_r_scale, dtype=float)
    sensor = MarkerSensor(sim.sensor_extrinsics()["camera"], noise)
    tracker = LQRTracker(Ts, mode=cfg.tracker_mode, limits=Limits(omega_max=cfg.omega_max))
    mapper = None
    if cfg.planner == "lidar":
        from src.planning.lidar_map import LidarMapper
        mapper = LidarMapper(sim.sensor_extrinsics()["lidar"])
    mission = DockingMission(ekf, sensor, tracker, MissionConfig(Ts=Ts, planner=cfg.planner, timeout=cfg.timeout),
                             slew=SlewRateLimiter() if cfg.slew else None, lidar_mapper=mapper)

    keys = ["t", "phase", "x", "y", "th", "xh", "yh", "thh", "Pxx", "Pxy", "Pyy", "Pxt", "Pyt", "Ptt", "v_cmd", "w_cmd",
            "v_odom", "w_odom", "xr", "yr", "thr", "detected", "accepted", "nis", "zx", "zy", "zyaw", "z_range",
            "z_view", "sx", "sy", "syaw", "frame_lost"]
    log = {k: [] for k in keys}
    impulses = sorted(cfg.impulses)
    odom, scan = None, None
    max_steps = int(cfg.timeout / Ts) + 50
    for k in range(max_steps):
        t = k * Ts
        image = cam.render()
        lost = cfg.marker_dropout > 0 and rng.random() < cfg.marker_dropout
        cmd = mission.step(t, odom, None if lost else image, scan if cfg.lidar else None)
        info = mission.info
        est = info["est"]
        p = true_pose(sim)
        meas = info["meas"]
        ref = info["ref_pose"]
        row = dict(t=t, phase=info["phase"], x=p[0], y=p[1], th=p[2], xh=est.x_hat[0], yh=est.x_hat[1],
                   thh=est.x_hat[2], Pxx=est.P[0, 0], Pxy=est.P[0, 1], Pyy=est.P[1, 1], Pxt=est.P[0, 2],
                   Pyt=est.P[1, 2], Ptt=est.P[2, 2], v_cmd=cmd.v, w_cmd=cmd.omega,
                   v_odom=odom.v if odom else np.nan, w_odom=odom.omega if odom else np.nan,
                   xr=ref[0] if ref is not None else np.nan, yr=ref[1] if ref is not None else np.nan,
                   thr=ref[2] if ref is not None else np.nan, detected=int(meas is not None),
                   accepted=int(info["accepted"]), nis=info["nis"],
                   zx=meas.base_T_marker[0] if meas is not None else np.nan,
                   zy=meas.base_T_marker[1] if meas is not None else np.nan,
                   zyaw=meas.base_T_marker[2] if meas is not None else np.nan,
                   z_range=meas.range_bearing[0] if meas is not None else np.nan,
                   z_view=mission.sensor.last_planar.view_angle if meas is not None else np.nan,
                   sx=np.sqrt(meas.R[0, 0]) if meas is not None else np.nan,
                   sy=np.sqrt(meas.R[1, 1]) if meas is not None else np.nan,
                   syaw=np.sqrt(meas.R[2, 2]) if meas is not None else np.nan, frame_lost=int(lost))
        for kk in keys:
            log[kk].append(row[kk])
        if frame_hook is not None:
            frame_hook(k, image, mission, log)
        if mission.phase in (Phase.DONE, Phase.FAILED):
            # Let the robot come to rest, then log the final state.
            for _ in range(10):
                sim.step(cmd)
            break
        while impulses and impulses[0][0] <= t:
            _, dv, dw = impulses.pop(0)
            _apply_impulse(sim, dv, dw)
        obs = sim.step(cmd)
        odom, scan = obs.odom, obs.lidar
        if step_hook is not None:
            step_hook(sim, mission)
    cam.close()

    final = true_pose(sim)
    fe = dock_frame_error(final)
    fe_est = dock_frame_error(mission.ekf.estimate().x_hat)
    tol = DockTolerance()
    ok = (mission.phase == Phase.DONE and abs(fe[0]) <= tol.longitudinal and abs(fe[1]) <= tol.lateral
          and abs(fe[2]) <= tol.heading)
    status = "success" if ok else ("out_of_tolerance" if mission.phase == Phase.DONE else mission.phase.value)
    out = {k: np.array(v) if k != "phase" else np.array(v, dtype=object) for k, v in log.items()}
    out["final_true"] = final
    out["start_true"] = nominal + offset
    out["ekf_updates"] = mission.ekf.updates
    out["reinit_events"] = mission.ekf.reinit_events
    return EpisodeResult(cfg, out, mission.events, mission.plans, bool(ok), status, fe, fe_est,
                         float(log["t"][-1]) if log["t"] else 0.0)

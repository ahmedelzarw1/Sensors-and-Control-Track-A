"""Run one docking episode in MuJoCo and log everything, including ground truth for experiments.

The separation that matters: the mission (robot_code/robot_brain.py) receives only odometry, LiDAR scans and camera images,
plus what it is told: the start point and the approximate station position (layout.dock_hint_y). This file additionally
reads the true pose through experiments.true_position, and the true station position from the layout, but only to log
them next to the estimates and to score the docking. Neither flows back into the mission.

Condition parameters (start point, odometry noise, wheel scale mismatch, floor friction, velocity impulses, marker
dropout) exist so experiments/run_trials.py can vary them. Each one changes the simulated world or the sensor stream,
never the pipeline's knowledge of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import mujoco
import numpy as np

from experiments.true_position import true_pose
from robot_code import room_map as km
from robot_code.lqr_controller import Q_DEFAULT, R_DEFAULT, Limits, LQRTracker
from robot_code.lqr_controller import SlewRateLimiter
from robot_code.docking import DockTolerance, dock_frame_error
from robot_code.kalman_filter import EKF, EKFParams
from robot_code.robot_brain import DockingMission, MissionConfig, Phase
from robot_code.simulator import SimCamera
from robot_code.simulator import LidarSpec, OdomNoise, TB3Sim, load_room_sim, wheel_collision_geom
from robot_code.data_types import wrap
from robot_code.vision import MarkerSensor
from robot_code.vision import MarkerNoiseModel


@dataclass
class EpisodeConfig:
    start: str = "west_centre"                  # start point of the fixed demo room (used when layout is None)
    start_offset: Optional[np.ndarray] = None   # true start = nominal + offset. None: sampled from the prior
    seed: int = 0
    odom_noise: OdomNoise = field(default_factory=OdomNoise)
    ekf: Optional[EKFParams] = None             # None: calibrated and tuned parameters from settings/
    tracker_mode: str = "scheduled"
    # Operating limit on the commanded turn rate. None: the fastest turn the wheels allow, 2 r w_max / L = 2.75 rad/s
    # with the ROBOTIS wheel limit of 6.67 rad/s (the datasheet gives 2.84). Fast turn reversals at this rate gave at
    # most 2.3 degrees of heading error against odometry on the patched model (README.md, Robot model).
    omega_max: Optional[float] = None
    slew: bool = True
    wheel_scale: tuple = (1.0, 1.0)             # true wheel radius / radius the software believes (left, right)
    friction_scale: float = 1.0                 # multiplies the sliding friction of the wheels and the floor
    impulses: tuple = ()                        # ((t, dv_forward m/s, dw rad/s), ...) applied to the true body
    servo_push: Optional[tuple] = None          # (s after visual servoing starts, dv m/s, dw rad/s): a push while docking
    start_offset_scale: float = 1.0             # true start error = this times the prior std the robot assumes
    lidar_noise_scale: float = 1.0              # true LiDAR noise = this times the datasheet noise the robot assumes
    lidar_fix: bool = True                      # False: the navigation EKF gets no LiDAR position fix (odometry only)
    lqr_r_scale: float = 1.0                    # multiplies the LQR input weight R (trade-off study)
    marker_dropout: float = 0.0                 # probability that a camera frame is lost
    timeout: float = 240.0
    lidar: bool = True                          # False: no LiDAR scans at all (odometry only until the end)
    vision_static_noise_only: bool = False      # True: ignore the measured in-motion inflation (used to measure it)
    vision_r_scale: tuple = (1.0, 1.0, 1.0)     # extra multiplier on the camera measurement std (x, y, yaw), for tuning
    use_tuning: bool = True                     # False: ignore settings/kalman_filter_tuning.json (used by the tuning itself)
    layout: Optional[km.Layout] = None          # None: the fixed demo room with start point `start`; else a (random) room


@dataclass
class EpisodeResult:
    cfg: EpisodeConfig
    layout: km.Layout
    log: dict
    events: list
    plans: list
    success: bool
    status: str
    final_error: np.ndarray      # dock-frame (longitudinal, lateral, heading) of the TRUE final pose
    final_est_error: np.ndarray  # the same for the docking EKF estimate at the end (marker frame)
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


def marker_to_room(pose_m, marker_w) -> np.ndarray:
    """A pose in the marker frame M expressed in the room frame W, given the marker pose in W (display only)."""
    pose_m = np.asarray(pose_m, dtype=float)
    c, s = np.cos(marker_w[2]), np.sin(marker_w[2])
    x = marker_w[0] + c * pose_m[..., 0] - s * pose_m[..., 1]
    y = marker_w[1] + s * pose_m[..., 0] + c * pose_m[..., 1]
    return np.stack([x, y, wrap(pose_m[..., 2] + marker_w[2])], axis=-1)


def room_to_marker(pose_w, marker_w) -> np.ndarray:
    """A pose in the room frame W expressed in the marker frame M."""
    c, s = np.cos(marker_w[2]), np.sin(marker_w[2])
    dx, dy = pose_w[0] - marker_w[0], pose_w[1] - marker_w[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(pose_w[2] - marker_w[2]))])


def robot_obstacle_contacts(sim: TB3Sim) -> list:
    """Names of the geoms the robot touches, other than the floor (an empty list means no clash)."""
    m, d = sim.model, sim.data
    robot = {sim._foot} | {b for b in range(m.nbody) if m.body_parentid[b] == sim._foot}  # noqa: SLF001
    out = []
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        for a, b in ((g1, g2), (g2, g1)):
            if m.geom_bodyid[a] in robot and m.geom_bodyid[b] not in robot:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b)
                if name != "floor":
                    out.append(name)
    return out


def run_episode(cfg: EpisodeConfig, frame_hook=None, step_hook: Optional[Callable] = None) -> EpisodeResult:
    """frame_hook(k, image, mission, log) is called every control step (video, photos, monitor); a list of hooks is
    allowed. step_hook(sim, mission) is called after every control step (used by the live viewer)."""
    hooks = [] if frame_hook is None else (list(frame_hook) if isinstance(frame_hook, (list, tuple)) else [frame_hook])
    rng = np.random.default_rng(cfg.seed)
    layout = cfg.layout if cfg.layout is not None else km.default_layout(cfg.start)
    nominal = layout.start.copy()
    offset = (rng.normal(0.0, cfg.start_offset_scale * km.START_PRIOR_SIGMA) if cfg.start_offset is None
              else np.asarray(cfg.start_offset))
    spec = LidarSpec.from_json()
    spec = replace(spec, sigma_near=spec.sigma_near * cfg.lidar_noise_scale,
                   sigma_far_frac=spec.sigma_far_frac * cfg.lidar_noise_scale)
    sim = load_room_sim(layout, seed=cfg.seed, odom_noise=cfg.odom_noise, odom_radius_scale=cfg.wheel_scale,
                        lidar_spec=spec)
    _apply_conditions(sim, cfg)
    sim.reset(nominal + offset)
    # Physical clash check: after placing the robot, MuJoCo must report no contact with any obstacle or wall.
    clashes = robot_obstacle_contacts(sim)
    if clashes:
        raise RuntimeError(f"robot placed in contact with {clashes}")
    cam = SimCamera(sim)
    Ts = sim.ctrl_dt
    marker_true = km.marker_pose(layout.dock_y)     # evaluation and display only

    ekf_params = cfg.ekf if cfg.ekf is not None else EKFParams.calibrated(use_tuning=cfg.use_tuning)
    ekf = EKF(nominal, np.diag(km.START_PRIOR_SIGMA ** 2), ekf_params)
    noise = MarkerNoiseModel.from_json(static_only=cfg.vision_static_noise_only, use_tuning=cfg.use_tuning)
    noise.inflation = noise.inflation * np.asarray(cfg.vision_r_scale, dtype=float)
    sensor = MarkerSensor(sim.sensor_extrinsics()["camera"], noise)
    p = sim.params
    w_lim = cfg.omega_max if cfg.omega_max is not None else 2.0 * p.wheel_radius * p.max_wheel_speed / p.wheel_separation
    tracker = LQRTracker(Ts, mode=cfg.tracker_mode, Q=Q_DEFAULT, R=R_DEFAULT * cfg.lqr_r_scale,
                         limits=Limits(v_max=p.wheel_radius * p.max_wheel_speed, omega_max=w_lim))
    lidar_xy = tuple(float(v) for v in sim.sensor_extrinsics()["lidar"]["t"][:2])
    mission = DockingMission(ekf, sensor, tracker, layout.dock_hint_y,
                             MissionConfig(Ts=Ts, timeout=cfg.timeout, lidar_offset_xy=lidar_xy,
                                           use_lidar_fix=cfg.lidar_fix),
                             slew=SlewRateLimiter() if cfg.slew else None)

    keys = ["t", "phase", "x", "y", "th", "xh", "yh", "thh", "Pxx", "Pxy", "Pyy", "Pxt", "Pyt", "Ptt", "v_cmd", "w_cmd",
            "v_odom", "w_odom", "xr", "yr", "thr", "detected", "accepted", "nis", "zx", "zy", "zyaw", "z_range",
            "z_view", "sx", "sy", "syaw", "frame_lost", "lidar_fix", "lidar_accepted", "lidar_nis",
            "mx", "my", "mth", "mxh", "myh", "mthh", "mPxx", "mPxy", "mPyy", "mPxt", "mPyt", "mPtt", "xd", "yd", "thd", "du_v", "du_w"]
    log = {k: [] for k in keys}
    log["marker_true"] = marker_true
    impulses = sorted(cfg.impulses)
    servo_start, contact_steps, contact_names = None, 0, set()
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
        if ref is not None and info["ref_frame"] == "marker":
            ref = marker_to_room(ref, marker_true)          # drawn in the room at the true marker (display only)
        pm = room_to_marker(p, marker_true)                 # true pose relative to the marker
        ed = info["est_dock"]
        dw = marker_to_room(ed.x_hat, marker_true) if ed is not None else np.full(3, np.nan)
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
                   syaw=np.sqrt(meas.R[2, 2]) if meas is not None else np.nan, frame_lost=int(lost),
                   lidar_fix=int(info["lidar_fix"] is not None), lidar_accepted=int(info["lidar_accepted"]),
                   lidar_nis=info["lidar_nis"], mx=pm[0], my=pm[1], mth=pm[2],
                   mxh=ed.x_hat[0] if ed is not None else np.nan, myh=ed.x_hat[1] if ed is not None else np.nan,
                   mthh=ed.x_hat[2] if ed is not None else np.nan, mPxx=ed.P[0, 0] if ed is not None else np.nan,
                   mPxy=ed.P[0, 1] if ed is not None else np.nan, mPyy=ed.P[1, 1] if ed is not None else np.nan,
                   mPxt=ed.P[0, 2] if ed is not None else np.nan, mPyt=ed.P[1, 2] if ed is not None else np.nan,
                   mPtt=ed.P[2, 2] if ed is not None else np.nan,
                   xd=dw[0], yd=dw[1], thd=dw[2],
                   du_v=info["du"][0] if info["du"] is not None else np.nan,
                   du_w=info["du"][1] if info["du"] is not None else np.nan)
        for kk in keys:
            log[kk].append(row[kk])
        for hook in hooks:
            hook(k, image, mission, log)
        if mission.phase in (Phase.DONE, Phase.FAILED):
            # Let the robot come to rest, then log the final state.
            for _ in range(10):
                sim.step(cmd)
            break
        while impulses and impulses[0][0] <= t:
            _, dv, dw = impulses.pop(0)
            _apply_impulse(sim, dv, dw)
        if servo_start is None and mission.phase == Phase.SERVO:
            servo_start = t
        if cfg.servo_push is not None and servo_start is not None and t >= servo_start + cfg.servo_push[0]:
            _apply_impulse(sim, cfg.servo_push[1], cfg.servo_push[2])
            cfg = replace(cfg, servo_push=None)   # once
        obs = sim.step(cmd)
        # Collision count (evaluation only): steps on which the robot touches an obstacle or a wall. The charging station
        # is excluded, the robot is meant to drive up to it.
        hit = [n for n in robot_obstacle_contacts(sim) if n not in ("charging_station", "aruco_plate")]
        if hit:
            contact_steps += 1
            contact_names.update(hit)
        odom, scan = obs.odom, obs.lidar
        if step_hook is not None:
            step_hook(sim, mission)
    cam.close()

    final = true_pose(sim)
    fe = dock_frame_error(final, km.dock_pose(layout.dock_y))
    fe_est = (dock_frame_error(mission.dock_ekf.estimate().x_hat) if mission.dock_ekf is not None
              else np.full(3, np.nan))
    tol = DockTolerance()
    ok = (mission.phase == Phase.DONE and abs(fe[0]) <= tol.longitudinal and abs(fe[1]) <= tol.lateral
          and abs(fe[2]) <= tol.heading)
    status = "success" if ok else ("out_of_tolerance" if mission.phase == Phase.DONE else mission.phase.value)
    out = {k: np.array(v) if k != "phase" else np.array(v, dtype=object) for k, v in log.items() if k in keys}
    out["marker_true"] = marker_true
    out["final_true"] = final
    out["start_true"] = nominal + offset
    out["start_prior"] = nominal
    out["contact_steps"] = contact_steps
    out["contact_names"] = sorted(contact_names)
    out["ekf_updates"] = mission.ekf.updates                                    # navigation EKF (LiDAR fixes)
    out["dock_updates"] = mission.dock_ekf.updates if mission.dock_ekf is not None else []   # docking EKF (camera)
    out["reinit_events"] = mission.dock_ekf.reinit_events if mission.dock_ekf is not None else []
    return EpisodeResult(cfg, layout, out, mission.events, mission.plans, bool(ok), status, fe, fe_est,
                         float(log["t"][-1]) if log["t"] else 0.0)

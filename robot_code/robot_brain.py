"""The docking mission: sensors in, velocity command out. This is the whole on-robot pipeline.

Given: the start point (the prior of the navigation EKF) and roughly where the charging station is (dock_hint_y).
Inputs each control period: wheel odometry, a LiDAR scan (every second step, 5 Hz) and the camera image. It never
receives ground truth.

Per step
1. Navigation EKF (room frame): predict with the odometry, update with the LiDAR position fix (the scan matched to the
   known walls, robot_code/lidar.py).
2. LiDAR: add the scan to the obstacle map, placed with the EKF estimate (collision detection and replanning).
3. Camera: used only at the end (phases LOOK and SERVO). The docking EKF (marker frame) predicts with the odometry
   and is updated with the marker measurement.
4. Phase logic below picks the command. 5. The slew-rate limiter smooths it.

Phases
- LOCALISE: hold still until the navigation EKF has accepted a LiDAR position fix (at most localise_timeout), so the
          first plan starts from a measured pose and not only from the start point.
- PLAN:   plan a path from the EKF estimate to the approach point in front of the given station position, on the
          LiDAR map (walls plus the obstacles found so far).
- ALIGN:  turn in place to a heading (state feedback on the estimated heading).
- TRACK:  LQR tracking of the path on the EKF estimate. Back to PLAN if the estimated tracking error becomes large, if
          the LiDAR shows an obstacle on the rest of the path, or if the LiDAR safety stop triggers (something too close
          ahead). At the end of the path: ALIGN to face the given station position, then LOOK.
- LOOK:   start the docking EKF from the navigation estimate and the given station position, then hold still (or
          search by turning) until the camera has given servo_fixes accepted marker measurements.
- SERVO:  visual servoing: LQR tracking of the docking reference built in the marker frame, on the docking EKF
          estimate; stop by the stop rule. If the marker is lost for dock_fix_timeout, back to LOOK.
- DONE / FAILED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from robot_code import room_map as km
from robot_code.lqr_controller import LQRTracker
from robot_code.lqr_controller import SlewRateLimiter
from robot_code.docking import (MarkerSearch, docking_reference, join_point, marker_frame_prior, remaining_distance,
                                should_stop)
from robot_code.kalman_filter import EKF
from robot_code.lidar import LidarMap, front_blocked, front_distance, wall_fix
from robot_code.unicycle_model import tracking_error
from robot_code.path_planning import PlannerParams, plan_reference
from robot_code.data_types import ControlCmd, LidarScan, Odometry, ReferenceTrajectory, StateEstimate, wrap


class Phase(str, Enum):
    LOCALISE = "localise"
    PLAN = "plan"
    ALIGN = "align"
    TRACK = "track"
    LOOK = "look"
    SERVO = "servo"
    DONE = "done"
    FAILED = "failed"


@dataclass
class MissionConfig:
    Ts: float = 0.1
    localise_timeout: float = 3.0     # s waiting for the first LiDAR fix, then plan on the start point alone
    align_tol: float = np.deg2rad(8.0)
    align_gain: float = 2.0
    align_omega: float = 1.0
    replan_pos: float = 0.25          # m, estimated tracking error that triggers a replan
    replan_heading: float = np.deg2rad(35.0)
    dock_align_max: float = np.deg2rad(60.0)  # larger heading error to the join point is fixed by turning first
    dock_fix_timeout: float = 3.0     # s without an accepted marker fix while servoing: stop and look again
    servo_fixes: int = 3              # accepted marker fixes before the visual servoing starts
    look_timeout: float = 30.0        # s looking for the marker before giving up
    timeout: float = 240.0            # s
    max_replans: int = 12
    planner_params: PlannerParams = field(default_factory=PlannerParams)
    lidar_offset_xy: tuple = (-0.032, 0.0)   # LiDAR position in the robot frame (ROBOTIS LDS-01 mount)
    lidar_replan_interval: float = 1.0       # s, at most one LiDAR-triggered replan per interval
    max_lidar_replans: int = 40
    use_lidar_fix: bool = True               # False: the navigation EKF runs on odometry alone (comparison only)


class DockingMission:
    def __init__(self, ekf: EKF, marker_sensor, tracker: LQRTracker, dock_hint_y: float,
                 cfg: MissionConfig = MissionConfig(), slew: Optional[SlewRateLimiter] = None):
        self.ekf, self.sensor, self.tracker, self.cfg = ekf, marker_sensor, tracker, cfg
        self.dock_hint_y = float(dock_hint_y)
        self.approach = km.approach_pose(self.dock_hint_y)        # end point of the trip (room frame)
        self.marker_hint = km.marker_pose(self.dock_hint_y)       # where the marker should be (room frame)
        self.dock_ekf: Optional[EKF] = None                       # created at the end, in the marker frame
        self.slew = slew
        if self.slew is not None:
            self.slew.reset()
        self.lidar_map = LidarMap(cfg.lidar_offset_xy)
        self.last_scan: Optional[LidarScan] = None
        self.new_scan = False
        self.last_lidar_replan = -np.inf
        self.n_lidar_replans = 0
        self.phase = Phase.LOCALISE
        self.ref: Optional[ReferenceTrajectory] = None
        self.ref_frame = "room"
        self.k = 0
        self.t = 0.0
        self.n_fixes = 0
        self.last_fix_t = -np.inf
        self.look_start = 0.0
        self.align_target = 0.0
        self.after_align = Phase.TRACK
        self.search = MarkerSearch()
        self.n_replans = 0
        self.events: list[tuple[float, str]] = []
        self.plans: list = []
        self.info: dict = {}

    # ------------------------------------------------------------------ helpers
    def _event(self, msg: str) -> None:
        self.events.append((round(self.t, 2), msg))

    def _plan(self, est: StateEstimate) -> None:
        plan = plan_reference(self.lidar_map.grid(), est.x_hat, self.approach, self.cfg.Ts, self.cfg.planner_params)
        if plan is None:
            self._event("no path found")
            self.phase = Phase.FAILED
            return
        self.plans.append((self.t, plan))
        self.ref, self.ref_frame, self.k = plan.ref, "room", 0
        self._start_align(float(plan.ref.theta[0]), Phase.TRACK)

    def _start_align(self, target: float, then: Phase) -> None:
        self.align_target, self.after_align = target, then
        self.phase = Phase.ALIGN

    def _start_look(self, est: StateEstimate) -> None:
        """Start the docking EKF in the marker frame from the navigation estimate and the given station position."""
        x_m, P_m = marker_frame_prior(est.x_hat, est.P, self.dock_hint_y)
        self.dock_ekf = EKF(x_m, P_m, self.ekf.p, marker_pose=np.zeros(3))
        self.dock_ekf.t = self.t
        self.n_fixes, self.last_fix_t, self.look_start = 0, -np.inf, self.t
        self.search = MarkerSearch()
        self.ref = None
        self.phase = Phase.LOOK
        self._event("at the approach point: camera on, looking for the marker")

    def _lidar_replan_needed(self) -> bool:
        """LiDAR checks while following a path: the safety stop, and a newly seen obstacle on the rest of the path.
        Either one sends the robot back to PLAN (at most once per lidar_replan_interval)."""
        if not self.new_scan or self.t - self.last_lidar_replan < self.cfg.lidar_replan_interval:
            return False
        reason = None
        if front_blocked(self.last_scan, self.cfg.lidar_offset_xy):
            reason = f"LiDAR safety stop: obstacle {front_distance(self.last_scan, self.cfg.lidar_offset_xy):.2f} m ahead"
        else:
            rest = slice(self.k, len(self.ref), 3)
            if self.lidar_map.path_blocked(self.ref.x[rest], self.ref.y[rest], km.ROBOT_RADIUS + 0.03):
                reason = "LiDAR: obstacle on the path"
        if reason is None:
            return False
        self.last_lidar_replan = self.t
        self.n_lidar_replans += 1
        self._event(reason + ", replanning")
        if self.n_lidar_replans > self.cfg.max_lidar_replans:
            self._event("too many LiDAR replans")
            self.phase = Phase.FAILED
        else:
            self.phase = Phase.PLAN
        return True

    # ------------------------------------------------------------------ main step
    def step(self, t: float, odom: Optional[Odometry], image: Optional[np.ndarray],
             lidar: Optional[LidarScan] = None) -> ControlCmd:
        self.t = t
        # 1. Navigation EKF: odometry predict, LiDAR position fix update.
        if odom is not None:
            self.ekf.predict(odom)
            if self.dock_ekf is not None:
                self.dock_ekf.predict(odom)
        fix, fix_ok = None, False
        self.new_scan = lidar is not None
        if lidar is not None:
            self.last_scan = lidar
        if lidar is not None and self.cfg.use_lidar_fix:
            fix = wall_fix(lidar, self.ekf.x, self.dock_hint_y, self.cfg.lidar_offset_xy, self.ekf.p.lidar_r_scale)
            fix_ok = self.ekf.update_lidar(fix) if fix is not None else False
        est = self.ekf.estimate()
        # 2. LiDAR obstacle map, placed with the estimate.
        if lidar is not None:
            self.lidar_map.add_scan(lidar, est)
        # 3. Camera, only at the end of the trip.
        meas, accepted = None, False
        if self.phase in (Phase.LOOK, Phase.SERVO) and image is not None:
            meas = self.sensor.measure(image, t)
            accepted = self.dock_ekf.update_marker(meas) if meas is not None else False
            if accepted:
                self.last_fix_t = t
                self.n_fixes += 1
        cam_nis = self.dock_ekf.updates[-1].nis if meas is not None and self.dock_ekf.updates else np.nan
        self.info = {"meas": meas, "accepted": accepted, "nis": cam_nis,
                     "lidar_fix": fix, "lidar_accepted": fix_ok,
                     "lidar_nis": self.ekf.updates[-1].nis if fix is not None else np.nan,
                     "front": front_distance(self.last_scan, self.cfg.lidar_offset_xy) if self.last_scan else np.inf}

        # 4. Phase logic, 5. slew-rate limit.
        self.tracker.last_du = None
        cmd = self._decide(est)
        if self.phase in (Phase.DONE, Phase.FAILED):
            cmd = ControlCmd(0.0, 0.0)   # the stop is not slew limited
            if self.slew is not None:
                self.slew.reset()
        elif self.slew is not None:
            cmd = self.slew.apply(cmd)
        self.info.update(phase=self.phase.value, est=est, cmd=cmd,
                         est_dock=self.dock_ekf.estimate() if self.dock_ekf is not None else None,
                         du=self.tracker.last_du,
                         ref_pose=self.ref.pose(self.k) if self.ref is not None else None, ref_frame=self.ref_frame)
        return cmd

    def _decide(self, est: StateEstimate) -> ControlCmd:
        if self.phase in (Phase.DONE, Phase.FAILED):
            return ControlCmd(0.0, 0.0)
        if self.t > self.cfg.timeout:
            self._event("timeout")
            self.phase = Phase.FAILED
            return ControlCmd(0.0, 0.0)

        if self.phase == Phase.LOCALISE:
            if self.info["lidar_accepted"]:
                self._event("localised from a LiDAR position fix")
                self.phase = Phase.PLAN
            elif self.t >= self.cfg.localise_timeout or not self.cfg.use_lidar_fix:
                self._event("no LiDAR fix accepted yet, planning from the start point")
                self.phase = Phase.PLAN
            else:
                return ControlCmd(0.0, 0.0)

        if self.phase == Phase.PLAN:
            self._plan(est)
            if self.phase == Phase.FAILED:
                return ControlCmd(0.0, 0.0)

        if self.phase == Phase.ALIGN:
            err = float(wrap(self.align_target - est.x_hat[2]))
            if abs(err) > self.cfg.align_tol:
                return ControlCmd(0.0, float(np.clip(self.cfg.align_gain * err, -self.cfg.align_omega,
                                                     self.cfg.align_omega)))
            if self.after_align == Phase.LOOK:
                self._start_look(est)
            else:
                self.phase, self.k = self.after_align, 0

        if self.phase == Phase.TRACK:
            if self.k >= len(self.ref) - 1:
                # End point reached: face the given station position, then the camera takes over.
                mx, my = self.marker_hint[:2]
                self._start_align(float(np.arctan2(my - est.x_hat[1], mx - est.x_hat[0])), Phase.LOOK)
                return self._decide(est)
            e = tracking_error(est.x_hat, self.ref.pose(self.k))
            if np.hypot(e[0], e[1]) > self.cfg.replan_pos or abs(e[2]) > self.cfg.replan_heading:
                self.n_replans += 1
                self._event(f"replan (estimated tracking error {np.hypot(e[0], e[1]):.2f} m, "
                            f"{np.rad2deg(e[2]):.0f} deg)")
                if self.n_replans > self.cfg.max_replans:
                    self.phase = Phase.FAILED
                    return ControlCmd(0.0, 0.0)
                self.phase = Phase.PLAN
                return self._decide(est)
            if self._lidar_replan_needed():
                return self._decide(est)
            cmd = self.tracker.command(est, self.ref, self.k)
            self.k += 1
            return cmd

        if self.phase == Phase.LOOK:
            est_m = self.dock_ekf.estimate()
            if self.n_fixes >= self.cfg.servo_fixes:
                self._event(f"marker found ({self.n_fixes} camera fixes), visual servoing")
                self.phase = Phase.SERVO
            elif self.t - self.look_start > self.cfg.look_timeout:
                self._event("marker not found")
                self.phase = Phase.FAILED
                return ControlCmd(0.0, 0.0)
            elif self.t - self.last_fix_t < 0.05:
                return ControlCmd(0.0, 0.0)   # marker in view: hold still while the fixes accumulate
            else:
                return self.search.command(est_m, (0.0, 0.0), self.cfg.Ts)

        if self.phase == Phase.SERVO:
            est_m = self.dock_ekf.estimate()
            if self.t - self.last_fix_t > self.cfg.dock_fix_timeout:
                self._event("marker lost while servoing, looking again")
                self.n_fixes, self.look_start, self.ref = 0, self.t, None
                self.search = MarkerSearch()
                self.phase = Phase.LOOK
                return self.search.command(est_m, (0.0, 0.0), self.cfg.Ts)
            if self.ref is None:
                # Turn first if the join point is far off to the side, then build the docking reference.
                jx, jy = join_point()
                err = float(wrap(np.arctan2(jy - est_m.x_hat[1], jx - est_m.x_hat[0]) - est_m.x_hat[2]))
                if abs(err) > self.cfg.dock_align_max:
                    return ControlCmd(0.0, float(np.clip(self.cfg.align_gain * err, -self.cfg.align_omega,
                                                         self.cfg.align_omega)))
                self.ref, self.ref_frame, self.k = docking_reference(est_m.x_hat, self.cfg.Ts), "marker", 0
                self._event("docking reference built in the marker frame")
            e = tracking_error(est_m.x_hat, self.ref.pose(self.k))
            if np.hypot(e[0], e[1]) > self.cfg.replan_pos or abs(e[2]) > self.cfg.replan_heading:
                self._event("docking reference rebuilt after a large estimated error")
                self.ref, self.k = docking_reference(est_m.x_hat, self.cfg.Ts), 0
            cmd = self.tracker.command(est_m, self.ref, self.k)
            self.k += 1
            if should_stop(est_m, cmd.v, self.cfg.Ts) or self.k >= len(self.ref):
                self._event(f"stop rule: estimated distance to go {remaining_distance(est_m):.3f} m")
                self.phase = Phase.DONE
                return ControlCmd(0.0, 0.0)
            return cmd
        return ControlCmd(0.0, 0.0)

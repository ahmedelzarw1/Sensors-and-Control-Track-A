"""The docking mission: sensors in, velocity command out. This is the whole on-robot pipeline.

Inputs each control period: wheel odometry, an optional LiDAR scan and the camera image. It never receives ground
truth. Plain-language walk-through: docs/pipeline.md.

Per step
1. EKF predict with the odometry, then EKF update with the marker measurement from the camera, if any.
2. LiDAR occupancy map update with the estimated pose (when the LiDAR planner is used).
3. Phase logic below picks the command. 4. The slew-rate limiter smooths it.

Phases
- LOCALISE: before driving, turn towards where the prior says the marker is (and sweep if needed) until
           localise_fixes marker measurements have been accepted, or give up after localise_timeout and continue on the
           prior alone. This initialises the estimate from the marker instead of only from the start pad.
- PLAN:    plan a path from the estimated pose to the pre-dock pose (known-map grid or LiDAR grid).
- ALIGN:   turn in place towards the start direction of the path (state feedback on the estimated heading).
- TRACK:   LQR tracking of the path. If the estimated tracking error becomes large (typically after the marker comes
           into view and corrects the estimate), go back to PLAN.
- ACQUIRE: at the end of the path, if the marker has not been accepted recently, search for it (src/docking).
- DOCK:    LQR tracking of the docking reference built from the current estimate; stop by the stop rule.
- DONE / FAILED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src import known_map as km
from src.control.lqr import LQRTracker
from src.control.slew import SlewRateLimiter
from src.docking.docking import MarkerSearch, docking_reference, remaining_distance, should_stop
from src.model.unicycle import tracking_error
from src.planning.grid import Grid, known_map_grid
from src.planning.planner import PlannerParams, plan_reference
from src.tb3.interfaces import ControlCmd, LidarScan, Odometry, ReferenceTrajectory, StateEstimate, wrap


class Phase(str, Enum):
    LOCALISE = "localise"
    PLAN = "plan"
    ALIGN = "align"
    TRACK = "track"
    ACQUIRE = "acquire"
    DOCK = "dock"
    DONE = "done"
    FAILED = "failed"


@dataclass
class MissionConfig:
    Ts: float = 0.1
    align_tol: float = np.deg2rad(8.0)
    align_gain: float = 2.0
    align_omega: float = 1.0
    replan_pos: float = 0.25          # m, estimated tracking error that triggers a replan
    replan_heading: float = np.deg2rad(35.0)
    fix_recent: float = 1.0           # s, a marker fix this recent counts as "marker in view"
    dock_align_max: float = np.deg2rad(60.0)  # larger heading error to the join point is fixed by turning first
    dock_fix_timeout: float = 3.0     # s without an accepted marker fix during docking: stop and search again
    timeout: float = 240.0            # s
    max_replans: int = 12
    planner: str = "known_map"        # "known_map" or "lidar"
    localise_fixes: int = 3           # accepted marker fixes required before driving
    localise_timeout: float = 20.0    # s, then continue on the prior
    planner_params: PlannerParams = field(default_factory=PlannerParams)


class DockingMission:
    def __init__(self, ekf, marker_sensor, tracker: LQRTracker, cfg: MissionConfig = MissionConfig(),
                 slew: Optional[SlewRateLimiter] = None, lidar_mapper=None):
        self.ekf, self.sensor, self.tracker, self.cfg = ekf, marker_sensor, tracker, cfg
        self.slew = slew
        if self.slew is not None:
            self.slew.reset()
        self.mapper = lidar_mapper
        self.known_grid = known_map_grid()
        self.phase = Phase.LOCALISE if cfg.localise_fixes > 0 else Phase.PLAN
        self.n_fixes = 0
        self.ref: Optional[ReferenceTrajectory] = None
        self.k = 0
        self.t = 0.0
        self.last_fix_t = -np.inf
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

    def planning_grid(self) -> Grid:
        if self.cfg.planner == "lidar" and self.mapper is not None:
            return self.mapper.grid()
        return self.known_grid

    def _plan(self, est: StateEstimate) -> None:
        grid = self.planning_grid()
        plan = plan_reference(grid, est.x_hat, km.PRE_DOCK_POSE, self.cfg.Ts, self.cfg.planner_params)
        if plan is None and grid is not self.known_grid:
            self._event("lidar grid gave no path, falling back to the known map")
            plan = plan_reference(self.known_grid, est.x_hat, km.PRE_DOCK_POSE, self.cfg.Ts, self.cfg.planner_params)
        if plan is None:
            self._event("no path found")
            self.phase = Phase.FAILED
            return
        self.plans.append((self.t, plan))
        self.ref, self.k = plan.ref, 0
        self._start_align(float(plan.ref.theta[0]), Phase.TRACK)

    def _start_align(self, target: float, then: Phase) -> None:
        self.align_target, self.after_align = target, then
        self.phase = Phase.ALIGN

    def _start_dock(self, est: StateEstimate) -> None:
        join = km.DOCK_POSE[:2] - 0.30 * np.array([np.cos(km.DOCK_POSE[2]), np.sin(km.DOCK_POSE[2])])
        to_join = float(np.arctan2(join[1] - est.x_hat[1], join[0] - est.x_hat[0]))
        if abs(wrap(to_join - est.x_hat[2])) > self.cfg.dock_align_max:
            self._start_align(to_join, Phase.DOCK)
            return
        self.ref, self.k = docking_reference(est.x_hat, self.cfg.Ts), 0
        self.phase = Phase.DOCK
        self._event("docking reference built")

    def marker_recent(self) -> bool:
        return self.t - self.last_fix_t <= self.cfg.fix_recent

    # ------------------------------------------------------------------ main step
    def step(self, t: float, odom: Optional[Odometry], image: Optional[np.ndarray],
             lidar: Optional[LidarScan] = None) -> ControlCmd:
        self.t = t
        if odom is not None:
            self.ekf.predict(odom)
        meas = self.sensor.measure(image, t) if image is not None else None
        accepted = self.ekf.update_marker(meas) if meas is not None else False
        if accepted:
            self.last_fix_t = t
            self.n_fixes += 1
        est = self.ekf.estimate()
        if lidar is not None and self.mapper is not None:
            self.mapper.add_scan(lidar, est)
        self.info = {"meas": meas, "accepted": accepted,
                     "nis": self.ekf.updates[-1].nis if meas is not None and self.ekf.updates else np.nan}

        cmd = self._decide(est)
        if self.phase in (Phase.DONE, Phase.FAILED):
            cmd = ControlCmd(0.0, 0.0)   # the stop is not slew limited
            if self.slew is not None:
                self.slew.reset()
        elif self.slew is not None:
            cmd = self.slew.apply(cmd)
        self.info.update(phase=self.phase.value, est=est, cmd=cmd,
                         ref_pose=self.ref.pose(self.k) if self.ref is not None else None)
        return cmd

    def _decide(self, est: StateEstimate) -> ControlCmd:
        if self.phase in (Phase.DONE, Phase.FAILED):
            return ControlCmd(0.0, 0.0)
        if self.t > self.cfg.timeout:
            self._event("timeout")
            self.phase = Phase.FAILED
            return ControlCmd(0.0, 0.0)

        if self.phase == Phase.LOCALISE:
            if self.n_fixes >= self.cfg.localise_fixes:
                self._event(f"localised from {self.n_fixes} marker fixes")
                self.phase = Phase.PLAN
            elif self.t >= self.cfg.localise_timeout:
                self._event("no marker fix before the localisation timeout, continuing on the start prior")
                self.phase = Phase.PLAN
            elif self.t - self.last_fix_t < 0.05:
                return ControlCmd(0.0, 0.0)   # marker in view: hold still while the fixes accumulate
            else:
                return self.search.command(est, self.cfg.Ts)

        if self.phase == Phase.PLAN:
            self._plan(est)
            if self.phase == Phase.FAILED:
                return ControlCmd(0.0, 0.0)

        if self.phase == Phase.ALIGN:
            err = float(wrap(self.align_target - est.x_hat[2]))
            if abs(err) > self.cfg.align_tol:
                return ControlCmd(0.0, float(np.clip(self.cfg.align_gain * err, -self.cfg.align_omega,
                                                     self.cfg.align_omega)))
            if self.after_align == Phase.DOCK:
                self._start_dock(est)
            else:
                self.phase, self.k = self.after_align, 0

        if self.phase == Phase.TRACK:
            if self.k >= len(self.ref) - 1:
                if self.marker_recent():
                    self._start_dock(est)
                else:
                    self._event("end of path without a recent marker fix, searching")
                    self.search = MarkerSearch()
                    self.phase = Phase.ACQUIRE
            else:
                e = tracking_error(est.x_hat, self.ref.pose(self.k))
                if (np.hypot(e[0], e[1]) > self.cfg.replan_pos or abs(e[2]) > self.cfg.replan_heading):
                    self.n_replans += 1
                    self._event(f"replan (estimated tracking error {np.hypot(e[0], e[1]):.2f} m, "
                                f"{np.rad2deg(e[2]):.0f} deg)")
                    if self.n_replans > self.cfg.max_replans:
                        self.phase = Phase.FAILED
                        return ControlCmd(0.0, 0.0)
                    self.phase = Phase.PLAN
                    return self._decide(est)
                cmd = self.tracker.command(est, self.ref, self.k)
                self.k += 1
                return cmd

        if self.phase == Phase.ACQUIRE:
            if self.t - self.last_fix_t < 0.05:  # accepted this very step
                self._event("marker acquired")
                self._start_dock(est)
            else:
                return self.search.command(est, self.cfg.Ts)

        if self.phase == Phase.ALIGN:   # entered from _start_dock
            return self._decide(est)

        if self.phase == Phase.DOCK:
            # The final approach relies on the marker. Without recent fixes the estimate may be far off and the stop
            # rule would stop the robot in the wrong place, so search for the marker again instead.
            if self.t - self.last_fix_t > self.cfg.dock_fix_timeout:
                self._event("no marker fix during docking, searching again")
                self.search = MarkerSearch()
                self.phase = Phase.ACQUIRE
                return self.search.command(est, self.cfg.Ts)
            e = tracking_error(est.x_hat, self.ref.pose(self.k))
            if np.hypot(e[0], e[1]) > self.cfg.replan_pos or abs(e[2]) > self.cfg.replan_heading:
                self._event("docking reference rebuilt after a large estimated error")
                self.ref, self.k = docking_reference(est.x_hat, self.cfg.Ts), 0
            cmd = self.tracker.command(est, self.ref, self.k)
            self.k += 1
            if should_stop(est, cmd.v, self.cfg.Ts) or self.k >= len(self.ref):
                self._event(f"stop rule: estimated distance to go {remaining_distance(est):.3f} m")
                self.phase = Phase.DONE
                return ControlCmd(0.0, 0.0)
            return cmd
        return ControlCmd(0.0, 0.0)

"""A serial-monitor style printout: the randomised layout, then what the robot calculates, live.

Like reading values over a serial port from a real robot, every line shows quantities the robot computes on board:
- the EKF estimate (x, y, heading) and its 1 sigma uncertainty,
- the commanded velocities (v, omega) from the LQR tracker,
- the marker measurement (range, bearing) and whether the filter accepted it (NIS against the gate),
- the estimated tracking error against the planned reference.
The last column, "true err", is the distance between the estimate and the true pose. The robot never sees it; it is
printed by the evaluation code so the viewer can judge the estimate. Events (phase changes, replans, stop rule) are
printed as they happen.
"""
from __future__ import annotations

import sys

import numpy as np

from src import known_map as km
from src.model.unicycle import tracking_error


def print_layout(layout: km.Layout, report=None, out=sys.stdout) -> None:
    w = out.write
    w("=" * 104 + "\n")
    w(f"RANDOMISED LAYOUT  {layout.name}" + (f"   (seed {layout.seed})" if layout.seed is not None else "") + "\n")
    w("-" * 104 + "\n")
    w(f"{'obstacle':<20}{'type':<10}{'x [m]':>8}{'y [m]':>8}{'size [m]':>22}{'yaw [deg]':>11}{'bound r [m]':>13}\n")
    for ob in layout.obstacles:
        if isinstance(ob, km.Box):
            size, kind, yaw = f"{2 * ob.hx:.2f} x {2 * ob.hy:.2f}", "box", f"{np.rad2deg(ob.yaw):.0f}"
        else:
            size, kind, yaw = f"diameter {2 * ob.radius:.2f}", "cylinder", "-"
        w(f"{ob.name:<20}{kind:<10}{ob.x:>8.2f}{ob.y:>8.2f}{size:>22}{yaw:>11}{ob.bounding_radius:>13.2f}\n")
    s = layout.start
    w(f"start pad (robot's prior): x {s[0]:.2f} m, y {s[1]:.2f} m, heading {np.rad2deg(s[2]):.0f} deg\n")
    if report is not None:
        w(f"checks passed after {report.tries} draw(s): smallest gap between obstacles {report.min_obstacle_gap:.2f} m "
          f"(rule >= 0.40), to walls {report.min_wall_gap:.2f} m (rule >= 0.40)\n")
        w(f"  start clearance {report.start_clearance:.2f} m from obstacles, {report.start_wall_clearance:.2f} m from "
          f"walls; marker {report.start_marker_distance:.2f} m away, "
          f"{'in view' if report.marker_visible_from_start else 'NOT in view'}; planned path {report.path_length:.2f} m\n")
    w("=" * 104 + "\n")


class SerialMonitor:
    """Frame hook for evaluation.runner.run_episode that prints a line every `every` control steps."""

    HEADER = (f"{'t[s]':>6} {'phase':<9}{'x^[m]':>7}{'y^[m]':>7}{'th^[deg]':>9}{'sd_xy[mm]':>10}{'sd_th[deg]':>11}"
              f"{'v[m/s]':>8}{'w[rad/s]':>9}{'marker r[m]':>12}{'b[deg]':>7}{'NIS':>7}{'used':>5}"
              f"{'e_y[mm]':>8}{'true err[mm]':>13}")

    def __init__(self, every: int = 5, out=sys.stdout, true_error: bool = True):
        self.every, self.out, self.true_error = every, out, true_error
        self._n_events = 0
        self._started = False

    def __call__(self, k, image, mission, log):
        w = self.out.write
        if not self._started:
            w(self.HEADER + "\n")
            self._started = True
        while self._n_events < len(mission.events):
            t, msg = mission.events[self._n_events]
            w(f"{t:6.1f} >> {msg}\n")
            self._n_events += 1
        if k % self.every:
            return
        est, cmd, meas = mission.info["est"], mission.info["cmd"], mission.info["meas"]
        sd_xy = 1000 * np.sqrt(max(est.P[0, 0] + est.P[1, 1], 0.0))
        sd_th = np.rad2deg(np.sqrt(max(est.P[2, 2], 0.0)))
        if meas is not None:
            r, b = meas.range_bearing
            nis = mission.info["nis"]
            mtxt = f"{r:12.2f}{np.rad2deg(b):7.1f}{nis:7.2f}{'yes' if mission.info['accepted'] else 'no':>5}"
        else:
            mtxt = f"{'-':>12}{'-':>7}{'-':>7}{'':>5}"
        ref = mission.info["ref_pose"]
        ey = f"{1000 * tracking_error(est.x_hat, ref)[1]:8.1f}" if ref is not None else f"{'-':>8}"
        terr = (f"{1000 * np.hypot(log['x'][-1] - est.x_hat[0], log['y'][-1] - est.x_hat[1]):13.1f}"
                if self.true_error else "")
        w(f"{log['t'][-1]:6.1f} {mission.info['phase']:<9}{est.x_hat[0]:7.3f}{est.x_hat[1]:7.3f}"
          f"{np.rad2deg(est.x_hat[2]):9.1f}{sd_xy:10.1f}{sd_th:11.2f}{cmd.v:8.3f}{cmd.omega:9.3f}{mtxt}{ey}{terr}\n")
        self.out.flush()

"""Command slew-rate limiter.

Limits how fast v and omega may change per control period, so the wheel motors are not asked for sudden jumps in speed
(gentler on a real robot's motors and tyres). The limits are design choices to be evaluated (Stage 3 compares the
tracker with and without the limiter).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_code.data_types import ControlCmd


@dataclass
class SlewRateLimiter:
    dv_max: float = 0.05      # m/s per 0.1 s step, i.e. 0.5 m/s^2 (assumption)
    dw_max: float = 0.30      # rad/s per 0.1 s step, i.e. 3 rad/s^2 (assumption)

    def __post_init__(self):
        self._last = ControlCmd(0.0, 0.0)

    def reset(self) -> None:
        self._last = ControlCmd(0.0, 0.0)

    def apply(self, cmd: ControlCmd) -> ControlCmd:
        v = self._last.v + float(np.clip(cmd.v - self._last.v, -self.dv_max, self.dv_max))
        w = self._last.omega + float(np.clip(cmd.omega - self._last.omega, -self.dw_max, self.dw_max))
        self._last = ControlCmd(v, w)
        return self._last

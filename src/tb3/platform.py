"""MuJoCo TurtleBot3 Burger platform wrapper.

What this module does
- Loads the MJCF and reads the robot geometry from it (nothing is hard-coded).
- Accepts body velocity commands (v, omega) and converts them to wheel speeds.
- Returns wheel-encoder odometry and 2D LiDAR scans (ray casting, LDS-01 spec).

What it does not do
- It does not expose ground truth to the pipeline. The true pose is available only
  through the private method _ground_truth_pose(), which is read by evaluation/
  and by nothing else. tests/test_structure.py enforces this.
- It does not render the camera or place the ArUco marker. Those belong to the
  vision part of the team and are added in the scene file.

Noise values below are simulation settings chosen by the team, not measurements.
They are constructor arguments so that evaluation can sweep them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from .interfaces import ControlCmd, LidarScan, Observation, Odometry, wrap

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party" / "hakoniwa-mujoco-robots"
DEFAULT_XML = REPO_ROOT / "models" / "track_a_scene.xml"  # platform characterisation scene (open floor)
ROOM_XML = REPO_ROOT / "models" / "track_a_room.xml"       # Track A room with obstacles, dock and marker
LDS01_JSON = THIRD_PARTY / "config" / "sensors" / "lidar" / "lds-01.json"


@dataclass(frozen=True)
class TB3Params:
    """Geometry and limits read from the MJCF."""
    wheel_radius: float        # m
    wheel_separation: float    # m
    max_wheel_speed: float     # rad/s, from the actuator control range

    @classmethod
    def from_model(cls, m: mujoco.MjModel) -> "TB3Params":
        bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_left_link")
        br = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_right_link")
        # The wheel geom is the single geom attached to each wheel body: size[0] is the radius.
        gl = int(np.flatnonzero(m.geom_bodyid == bl)[0])
        radius = float(m.geom_size[gl][0])
        separation = float(abs(m.body_pos[bl][1] - m.body_pos[br][1]))
        al = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
        max_w = float(m.actuator_ctrlrange[al][1])
        return cls(radius, separation, max_w)


@dataclass(frozen=True)
class OdomNoise:
    """Additive Gaussian noise on the encoder-derived body velocities (assumed values)."""
    sigma_v: float = 0.005       # m/s
    sigma_omega: float = 0.01    # rad/s


@dataclass(frozen=True)
class LidarSpec:
    """LDS-01 specification read from the Hakoniwa JSON (range in mm there, metres here)."""
    r_min: float
    r_max: float
    sigma_near: float        # m, below near_limit
    near_limit: float        # m
    sigma_far_frac: float    # fraction of range, treated as 1 sigma (assumption, conservative)
    n_beams: int
    period: float            # s

    @classmethod
    def from_json(cls, path: Path = LDS01_JSON) -> "LidarSpec":
        spec = json.loads(Path(path).read_text())["spec"]
        acc = spec["DistanceAccuracy"]
        ang = spec["AngleRange"]
        n = int(round((ang["Max"] - ang["Min"]) / ang["Resolution"]))
        return cls(
            r_min=spec["DetectionDistance"]["Min"] / 1000.0,
            r_max=spec["DetectionDistance"]["Max"] / 1000.0,
            sigma_near=float(acc[0]["DistanceIndependentAccuracy"]["StdDev"]),
            near_limit=acc[0]["Range"]["Max"] / 1000.0,
            sigma_far_frac=float(acc[1]["DistanceDependentAccuracy"]["Percentage"]) / 100.0,
            n_beams=n,
            period=1.0 / float(ang["ScanFrequency"]),
        )

    def sigma(self, r: np.ndarray) -> np.ndarray:
        return np.where(r < self.near_limit, self.sigma_near, self.sigma_far_frac * r)


class TB3Sim:
    """One robot in one MuJoCo scene, stepped at the control period."""

    def __init__(
        self,
        xml_path: Path | str = DEFAULT_XML,
        ctrl_dt: float = 0.1,
        seed: int = 0,
        odom_noise: OdomNoise = OdomNoise(),
        lidar_spec: Optional[LidarSpec] = None,
        lidar_noise: bool = True,
        wheel_accel: Optional[float] = 30.0,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.params = TB3Params.from_model(self.model)
        self.ctrl_dt = float(ctrl_dt)
        n_sub = self.ctrl_dt / self.model.opt.timestep
        if abs(n_sub - round(n_sub)) > 1e-9:
            raise ValueError("ctrl_dt must be a multiple of the MuJoCo timestep")
        self._n_sub = int(round(n_sub))
        self.odom_noise = odom_noise
        self.lidar = lidar_spec if lidar_spec is not None else LidarSpec.from_json()
        self.lidar_noise = lidar_noise
        # Limit on how fast the DIFFERENCE between the wheel speeds may change (rad/s^2), applied to the actuator
        # setpoints at every physics substep; the common (forward) part changes as before. Why (docs/platform_notes.md):
        # a step in the turning command kicks the body, so on an arc the true heading change was only 94 percent of
        # the odometry value and at the end of a turn the body briefly yawed the wrong way. Ramping the turning part
        # removes most of that. Ramping the forward part as well made the robot slide back further after a stop (a
        # contact artefact of the simulator), so it is not ramped. 30 rad/s^2 is our ASSUMPTION, the gentlest value
        # tested that removed the kick. None reproduces the original behaviour.
        self.wheel_accel = wheel_accel
        self._rng = np.random.default_rng(seed)

        name2id = lambda typ, n: mujoco.mj_name2id(self.model, typ, n)
        B, J, A = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
        self._foot = name2id(B, "base_footprint")
        self._link = name2id(B, "base_link")
        self._scan = name2id(B, "base_scan")
        self._act_l, self._act_r = name2id(A, "left_motor"), name2id(A, "right_motor")
        self._qadr_l = int(self.model.jnt_qposadr[name2id(J, "wheel_left_joint")])
        self._qadr_r = int(self.model.jnt_qposadr[name2id(J, "wheel_right_joint")])
        self._free_adr = int(self.model.jnt_qposadr[self.model.body_jntadr[self._foot]])
        self._lidar_angles = wrap(np.deg2rad(-180.0) + np.arange(self.lidar.n_beams) * (2 * np.pi / self.lidar.n_beams))
        self.reset()

    # ------------------------------------------------------------------ control
    def reset(self, pose=(0.0, 0.0, 0.0)) -> None:
        """Place the robot at pose = (x, y, yaw) and zero all velocities."""
        mujoco.mj_resetData(self.model, self.data)
        a = self._free_adr
        x, y, th = pose
        self.data.qpos[a + 0], self.data.qpos[a + 1] = x, y
        self.data.qpos[a + 3: a + 7] = [np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self._lidar_acc = 0.0
        self._phi_prev = np.array([self.data.qpos[self._qadr_l], self.data.qpos[self._qadr_r]])

    def wheel_speeds(self, cmd: ControlCmd) -> np.ndarray:
        """Differential-drive inverse kinematics, clipped to the actuator range. Returns (w_left, w_right)."""
        r, L, wmax = self.params.wheel_radius, self.params.wheel_separation, self.params.max_wheel_speed
        w_r = (cmd.v + 0.5 * L * cmd.omega) / r
        w_l = (cmd.v - 0.5 * L * cmd.omega) / r
        return np.clip([w_l, w_r], -wmax, wmax)

    def step(self, cmd: ControlCmd) -> Observation:
        """Hold the command for one control period and return what the sensors report."""
        w_l, w_r = self.wheel_speeds(cmd)
        common, diff_target = 0.5 * (w_l + w_r), 0.5 * (w_r - w_l)
        dmax = np.inf if self.wheel_accel is None else self.wheel_accel * self.model.opt.timestep
        diff = 0.5 * (self.data.ctrl[self._act_r] - self.data.ctrl[self._act_l])
        for _ in range(self._n_sub):
            diff += float(np.clip(diff_target - diff, -dmax, dmax))
            self.data.ctrl[self._act_l], self.data.ctrl[self._act_r] = common - diff, common + diff
            mujoco.mj_step(self.model, self.data)
        self.t += self.ctrl_dt

        phi = np.array([self.data.qpos[self._qadr_l], self.data.qpos[self._qadr_r]])
        dphi = phi - self._phi_prev
        self._phi_prev = phi
        r, L = self.params.wheel_radius, self.params.wheel_separation
        v = r * (dphi[1] + dphi[0]) / (2.0 * self.ctrl_dt)
        w = r * (dphi[1] - dphi[0]) / (L * self.ctrl_dt)
        v += self._rng.normal(0.0, self.odom_noise.sigma_v)
        w += self._rng.normal(0.0, self.odom_noise.sigma_omega)
        odom = Odometry(t=self.t, v=float(v), omega=float(w), dt=self.ctrl_dt)

        scan = None
        self._lidar_acc += self.ctrl_dt
        if self._lidar_acc >= self.lidar.period - 1e-9:
            self._lidar_acc -= self.lidar.period
            scan = self._scan_lidar()
        return Observation(odom=odom, lidar=scan, extras={"wheel_cmd": (float(w_l), float(w_r))})

    # ------------------------------------------------------------------ sensors
    def _scan_lidar(self) -> LidarScan:
        # In the original MJCF the caster has no collision geometry and the body rests pitched about
        # 4 degrees nose-up on its rear edge (see docs/platform_notes.md). A real LDS-01 scans a level
        # plane, so the beams are cast horizontally using the yaw of base_footprint only. This keeps
        # the scan independent of any residual body tilt.
        origin = self.data.xpos[self._scan].copy()
        Rf = self.data.xmat[self._foot].reshape(3, 3)
        yaw = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
        geomid = np.zeros(1, dtype=np.int32)
        ranges = np.full(self.lidar.n_beams, np.nan)
        for i, a in enumerate(self._lidar_angles):
            d = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            dist = mujoco.mj_ray(self.model, self.data, origin, d, None, 1, self._scan, geomid)
            if dist >= 0.0:
                ranges[i] = dist
        valid = np.isfinite(ranges) & (ranges >= self.lidar.r_min) & (ranges <= self.lidar.r_max)
        ranges[~valid] = np.nan
        if self.lidar_noise:
            ranges[valid] += self._rng.normal(0.0, 1.0, valid.sum()) * self.lidar.sigma(ranges[valid])
        return LidarScan(t=self.t, angles=self._lidar_angles.copy(), ranges=ranges)

    def sensor_extrinsics(self) -> dict:
        """Sensor poses in the base_footprint frame, computed from the model at the current state.

        Returns translation (3,) and rotation matrix (3,3) for the LiDAR origin and the colour camera.
        The MuJoCo camera looks along its own -z axis, with +x to the image right and +y up.
        """
        Rb = self.data.xmat[self._foot].reshape(3, 3)
        pb = self.data.xpos[self._foot]
        cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "color_camera")
        out = {
            "lidar": {"t": Rb.T @ (self.data.xpos[self._scan] - pb), "R": Rb.T @ self.data.xmat[self._scan].reshape(3, 3)},
            "camera": {"t": Rb.T @ (self.data.cam_xpos[cam] - pb), "R": Rb.T @ self.data.cam_xmat[cam].reshape(3, 3)},
        }
        return out

    # ------------------------------------------------------------ ground truth
    def _ground_truth_pose(self) -> np.ndarray:
        """True planar pose (x, y, yaw) of base_footprint in W. Evaluation only. Never call from src/."""
        R = self.data.xmat[self._foot].reshape(3, 3)
        p = self.data.xpos[self._foot]
        return np.array([p[0], p[1], float(np.arctan2(R[1, 0], R[0, 0]))])

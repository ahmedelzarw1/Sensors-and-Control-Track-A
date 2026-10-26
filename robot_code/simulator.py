"""The simulated TurtleBot3 Burger in MuJoCo (the official ROBOTIS model): everything on the simulator side.

1. Robot: wheel commands -> MuJoCo physics -> wheel-encoder odometry and LiDAR scans (class TB3Sim).
2. Camera: renders the robot's forward camera.
3. Room: builds the MuJoCo room around the robot for any layout.

The true position is available only through TB3Sim._ground_truth_pose(), which only experiments/ reads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import mujoco
import numpy as np
from robot_code.data_types import ControlCmd, LidarScan, Observation, Odometry, wrap
from robot_code.vision import CAMERA_FOVY_DEG, CAMERA_HEIGHT, CAMERA_WIDTH
import os
import tempfile
from robot_code import room_map as km


# ==============================================================================================================
# 1. ROBOT
# MuJoCo TurtleBot3 Burger platform wrapper.
#
# The robot is the official ROBOTIS TurtleBot3 Burger model (downloaded_robot_model/robotis_mujoco_menagerie), patched by
# commands/build_mujoco_files.py into mujoco_files/robot.xml. The base frame B is the body "base", whose origin is on the
# ground below the centre of the wheel axle.
#
# What this module does
# - Loads the MJCF and reads the robot geometry from it (nothing is hard-coded).
# - Accepts body velocity commands (v, omega) and converts them to wheel speeds.
# - Returns wheel-encoder odometry and 2D LiDAR scans (ray casting, LDS-01 spec).
#
# What it does not do
# - It does not expose ground truth to the pipeline. The true pose is available only
#   through the private method _ground_truth_pose(), which is read by experiments/true_position.py
#   and by nothing else.
# - It does not render the camera or place the ArUco marker. Those belong to the
#   vision part of the team and are added in the scene file.
#
# Noise values below are simulation settings chosen by the team, not measurements.
# They are constructor arguments so that evaluation can sweep them.
# ==============================================================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "mujoco_files" / "open_floor.xml"  # platform characterisation scene (open floor)
ROOM_XML = REPO_ROOT / "mujoco_files" / "room.xml"       # Track A room with obstacles, dock and marker
LDS01_JSON = REPO_ROOT / "settings" / "lidar_spec.json"                 # ROBOTIS LDS-01 specification


@dataclass(frozen=True)
class TB3Params:
    """Geometry and limits read from the MJCF."""
    wheel_radius: float        # m
    wheel_separation: float    # m
    max_wheel_speed: float     # rad/s, from the actuator control range

    @classmethod
    def from_model(cls, m: mujoco.MjModel) -> "TB3Params":
        bl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_left")
        br = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wheel_right")
        # The tyre collision geom is the wheel geom that collides (the other one is visual only). It is a mesh, so the
        # radius is read from its bounding box: the two largest half extents are the radius, the smallest is the width.
        gl = wheel_collision_geom(m, bl)
        radius = float(np.sort(m.geom_aabb[gl][3:])[-1])
        separation = float(abs(m.body_pos[bl][1] - m.body_pos[br][1]))
        al = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left")
        max_w = float(m.actuator_ctrlrange[al][1])
        return cls(round(radius, 4), separation, max_w)


def wheel_collision_geom(m: mujoco.MjModel, wheel_body: int) -> int:
    """The colliding geom of a wheel body."""
    return int([g for g in np.flatnonzero(m.geom_bodyid == wheel_body) if m.geom_contype[g] or m.geom_conaffinity[g]][0])


@dataclass(frozen=True)
class OdomNoise:
    """Additive Gaussian noise on the encoder-derived body velocities (assumed values)."""
    sigma_v: float = 0.005       # m/s
    sigma_omega: float = 0.01    # rad/s


@dataclass(frozen=True)
class LidarSpec:
    """ROBOTIS LDS-01 specification, read from settings/lidar_spec.json."""
    r_min: float
    r_max: float
    sigma_near: float        # m, below near_limit
    near_limit: float        # m
    sigma_far_frac: float    # fraction of range, treated as 1 sigma (assumption, conservative)
    n_beams: int
    period: float            # s

    @classmethod
    def from_json(cls, path: Path = LDS01_JSON) -> "LidarSpec":
        spec = json.loads(Path(path).read_text())
        return cls(
            r_min=float(spec["range_min_m"]),
            r_max=float(spec["range_max_m"]),
            sigma_near=float(spec["sigma_near_m"]),
            near_limit=float(spec["near_limit_m"]),
            sigma_far_frac=float(spec["sigma_far_fraction"]),
            n_beams=int(round(360.0 / spec["angular_resolution_deg"])),
            period=1.0 / float(spec["scan_rate_hz"]),
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
        odom_radius_scale: tuple = (1.0, 1.0),
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        # The robot's software believes the wheel radii are true_radius / scale. 1.0 means perfectly known radii;
        # evaluation uses other values to simulate a wheel radius mismatch.
        self.odom_radius_scale = np.asarray(odom_radius_scale, dtype=float)
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
        self._rng = np.random.default_rng(seed)

        name2id = lambda typ, n: mujoco.mj_name2id(self.model, typ, n)
        B, J, A = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
        # ROBOTIS model: body "base" has its origin on the ground below the wheel axle centre (frame B).
        self._foot = name2id(B, "base")
        self._scan = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "base_scan")
        self._act_l, self._act_r = name2id(A, "wheel_left"), name2id(A, "wheel_right")
        self._qadr_l = int(self.model.jnt_qposadr[name2id(J, "wheel_left")])
        self._qadr_r = int(self.model.jnt_qposadr[name2id(J, "wheel_right")])
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
        # The start pose also becomes the model's default pose, so a reset from outside (for example the Reset button of
        # the MuJoCo viewer) puts the robot back here and not at the world origin, which may be inside an obstacle.
        self.model.qpos0[a: a + 7] = self.data.qpos[a: a + 7]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self._lidar_acc = 0.0
        self._phi_prev = np.array([self.data.qpos[self._qadr_l], self.data.qpos[self._qadr_r]])

    def _believed_radii(self) -> np.ndarray:
        """Wheel radii (left, right) the robot's software uses for commands and odometry."""
        return self.params.wheel_radius / self.odom_radius_scale

    def wheel_speeds(self, cmd: ControlCmd) -> np.ndarray:
        """Differential-drive inverse kinematics, clipped to the actuator range. Returns (w_left, w_right)."""
        (r_l, r_r), L, wmax = self._believed_radii(), self.params.wheel_separation, self.params.max_wheel_speed
        w_r = (cmd.v + 0.5 * L * cmd.omega) / r_r
        w_l = (cmd.v - 0.5 * L * cmd.omega) / r_l
        return np.clip([w_l, w_r], -wmax, wmax)

    def step(self, cmd: ControlCmd) -> Observation:
        """Hold the command for one control period and return what the sensors report."""
        w_l, w_r = self.wheel_speeds(cmd)
        # The wheel speed targets go to the velocity servos, and MuJoCo simulates the physics for one control period.
        self.data.ctrl[self._act_l], self.data.ctrl[self._act_r] = w_l, w_r
        for _ in range(self._n_sub):
            mujoco.mj_step(self.model, self.data)
        self.t += self.ctrl_dt

        phi = np.array([self.data.qpos[self._qadr_l], self.data.qpos[self._qadr_r]])
        dphi = phi - self._phi_prev
        self._phi_prev = phi
        (r_l, r_r), L = self._believed_radii(), self.params.wheel_separation
        v = (r_r * dphi[1] + r_l * dphi[0]) / (2.0 * self.ctrl_dt)
        w = (r_r * dphi[1] - r_l * dphi[0]) / (L * self.ctrl_dt)
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
        # Beams are cast horizontally from the LDS-01 origin (site base_scan) using the yaw of the base only, as the
        # real sensor scans a plane. This keeps the scan independent of any small body tilt. The robot's own body is
        # excluded so the beams do not hit the LiDAR housing they start inside.
        origin = self.data.site_xpos[self._scan].copy()
        Rf = self.data.xmat[self._foot].reshape(3, 3)
        yaw = float(np.arctan2(Rf[1, 0], Rf[0, 0]))
        geomid = np.zeros(1, dtype=np.int32)
        ranges = np.full(self.lidar.n_beams, np.nan)
        for i, a in enumerate(self._lidar_angles):
            d = np.array([np.cos(yaw + a), np.sin(yaw + a), 0.0])
            dist = mujoco.mj_ray(self.model, self.data, origin, d, None, 1, self._foot, geomid)
            if dist >= 0.0:
                ranges[i] = dist
        valid = np.isfinite(ranges) & (ranges >= self.lidar.r_min) & (ranges <= self.lidar.r_max)
        ranges[~valid] = np.nan
        if self.lidar_noise:
            ranges[valid] += self._rng.normal(0.0, 1.0, valid.sum()) * self.lidar.sigma(ranges[valid])
        return LidarScan(t=self.t, angles=self._lidar_angles.copy(), ranges=ranges)

    def sensor_extrinsics(self) -> dict:
        """Sensor poses in the base frame, computed from the model at the current state.

        Returns translation (3,) and rotation matrix (3,3) for the LiDAR origin and the colour camera.
        The MuJoCo camera looks along its own -z axis, with +x to the image right and +y up.
        """
        Rb = self.data.xmat[self._foot].reshape(3, 3)
        pb = self.data.xpos[self._foot]
        cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "color_camera")
        out = {
            "lidar": {"t": Rb.T @ (self.data.site_xpos[self._scan] - pb),
                      "R": Rb.T @ self.data.site_xmat[self._scan].reshape(3, 3)},
            "camera": {"t": Rb.T @ (self.data.cam_xpos[cam] - pb), "R": Rb.T @ self.data.cam_xmat[cam].reshape(3, 3)},
        }
        return out

    # ------------------------------------------------------------ ground truth
    def _ground_truth_pose(self) -> np.ndarray:
        """True planar pose (x, y, yaw) of base in W. Evaluation only. Never call from robot_code/."""
        R = self.data.xmat[self._foot].reshape(3, 3)
        p = self.data.xpos[self._foot]
        return np.array([p[0], p[1], float(np.arctan2(R[1, 0], R[0, 0]))])


# ==============================================================================================================
# 2. CAMERA
# Simulated robot camera: offscreen rendering of the fixed forward camera `color_camera`.
#
# This is the simulator side of the camera (it produces images, like the LiDAR ray casting in platform.py). Image
# processing is in robot_code/vision/.
#
# Rendering backend. MuJoCo picks the OpenGL backend from the MUJOCO_GL environment variable. On macOS the default (CGL)
# works and was used for every result in this repository. On Linux without a display, set MUJOCO_GL=egl (GPU) or
# MUJOCO_GL=osmesa (software, slow) before Python starts. The Linux backends have not been tested by us.
# ==============================================================================================================

class SimCamera:
    def __init__(self, sim, name: str = "color_camera", width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        self._sim = sim
        self.name = name
        self.width, self.height = width, height
        cam = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam < 0:
            raise ValueError(f"camera {name!r} not in model")
        self.fovy_deg = float(sim.model.cam_fovy[cam])
        if abs(self.fovy_deg - CAMERA_FOVY_DEG) > 1e-6:
            raise ValueError(f"scene fovy {self.fovy_deg} differs from camera_model.CAMERA_FOVY_DEG; regenerate the scene")
        self._renderer = mujoco.Renderer(sim.model, height, width)

    def render(self) -> np.ndarray:
        """RGB image (H, W, 3) uint8 with row 0 at the top, as OpenCV expects."""
        self._renderer.update_scene(self._sim.data, camera=self.name)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


# ==============================================================================================================
# 3. ROOM
# MuJoCo XML for the Track A room around the patched ROBOTIS TurtleBot3 Burger (mujoco_files/robot.xml).
#
# room_scene_xml(layout) builds the room for any Layout (robot_code/room_map.py): walls, obstacles, charging station and
# ArUco marker, with the station at the layout's TRUE position. Every run builds its room on the fly with
# load_room_sim(). (mujoco_files/room.xml is the fixed demo room with the station at y = 0.40, used by the camera
# calibration.)
# ==============================================================================================================

MODELS = Path(__file__).resolve().parents[1] / "mujoco_files"
MARKER_TEXTURE = "textures/aruco_4x4_50_id0.png"   # relative to mujoco_files/
FLOOR_BITS = 'contype="2" conaffinity="2"'          # see commands/build_mujoco_files.py


def _obstacle_geom(ob) -> str:
    if isinstance(ob, km.Box):
        q = (np.cos(ob.yaw / 2), 0.0, 0.0, np.sin(ob.yaw / 2))
        return (f'    <geom name="{ob.name}" type="box" pos="{ob.x:g} {ob.y:g} {ob.height / 2:g}" '
                f'size="{ob.hx:g} {ob.hy:g} {ob.height / 2:g}" quat="{q[0]:.6g} 0 0 {q[3]:.6g}" '
                'rgba="0.85 0.45 0.25 1"/>')
    return (f'    <geom name="{ob.name}" type="cylinder" pos="{ob.x:g} {ob.y:g} {ob.height / 2:g}" '
            f'size="{ob.radius:g} {ob.height / 2:g}" rgba="0.25 0.55 0.85 1"/>')


def room_scene_xml(layout: Optional[km.Layout] = None) -> str:
    t, h = km.WALL_THICKNESS, km.WALL_HEIGHT
    x0, x1, y0, y1 = km.ROOM_X_MIN, km.ROOM_X_MAX, km.ROOM_Y_MIN, km.ROOM_Y_MAX
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hx, hy = 0.5 * (x1 - x0) + t, 0.5 * (y1 - y0)
    walls = [("wall_north", cx, y1 + t / 2, hx, t / 2), ("wall_south", cx, y0 - t / 2, hx, t / 2),
             ("wall_east", x1 + t / 2, cy, t / 2, hy), ("wall_west", x0 - t / 2, cy, t / 2, hy)]
    geoms = []
    for name, x, y, sx, sy in walls:
        geoms.append(f'    <geom name="{name}" type="box" pos="{x:g} {y:g} {h / 2:g}" size="{sx:g} {sy:g} {h / 2:g}" '
                     'material="wall_mat"/>')
    layout = layout if layout is not None else km.default_layout()
    for ob in layout.obstacles:
        geoms.append(_obstacle_geom(ob))
    sx = x1 - km.STATION_DEPTH / 2
    dock_y = layout.dock_y   # the TRUE station position (the robot is only told layout.dock_hint_y)
    geoms.append(f'    <geom name="charging_station" type="box" pos="{sx:g} {dock_y:g} {km.STATION_HEIGHT / 2:g}" '
                 f'size="{km.STATION_DEPTH / 2:g} {km.STATION_HALF_WIDTH:g} {km.STATION_HEIGHT / 2:g}" '
                 'material="station_mat"/>')
    plate = km.MARKER_SIZE / 2 + km.MARKER_WHITE_BORDER
    mx = km.STATION_FRONT_X - km.MARKER_PLATE_THICKNESS / 2
    # MuJoCo draws a 2d texture unmirrored on the box face whose normal is the local +z axis. Local +z is the marker
    # outward normal (world -x), local y is world +z (image up), local x is world -y (image right for a viewer).
    geoms.append(f'    <body name="aruco_marker" pos="{mx:g} {dock_y:g} {km.MARKER_Z:g}" xyaxes="0 -1 0 0 0 1">\n'
                 f'      <geom name="aruco_plate" type="box" size="{plate:g} {plate:g} '
                 f'{km.MARKER_PLATE_THICKNESS / 2:g}" material="aruco_mat" contype="0" conaffinity="0"/>\n'
                 '    </body>')
    rel = MARKER_TEXTURE
    # znear is a fraction of the model extent. The extent is fixed at 4 m so the near clip plane sits at 8 mm and the
    # marker is visible when docked (with the automatic extent the clip plane hid it).
    return f"""<mujoco model="Track A room">
  <include file="robot.xml"/>

  <statistic center="0 0 0.3" extent="4"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="90" elevation="-60" offwidth="{CAMERA_WIDTH}" offheight="{CAMERA_HEIGHT}"/>
    <map znear="0.002" zfar="20"/>
  </visual>

  <asset>
    <texture name="aruco_tex" type="2d" file="{rel}"/>
    <material name="aruco_mat" texture="aruco_tex" texrepeat="1 1" specular="0" shininess="0" emission="0.6"/>
    <texture name="floor_tex" type="2d" builtin="checker" width="256" height="256" rgb1="0.62 0.60 0.56"
      rgb2="0.56 0.54 0.50"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0"/>
    <material name="wall_mat" rgba="0.82 0.80 0.76 1"/>
    <material name="station_mat" rgba="0.15 0.15 0.18 1"/>
  </asset>

  <worldbody>
    <light name="ceiling" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat" {FLOOR_BITS}/>
""" + "\n".join(geoms) + """
  </worldbody>
</mujoco>
"""


def load_room_sim(layout: km.Layout, **sim_kwargs):
    """A TB3Sim in the room of the given layout. The XML is written next to mujoco_files/robot.xml (the include and
    texture paths are relative to mujoco_files/), compiled, and deleted again."""
    fd, path = tempfile.mkstemp(prefix=".layout_", suffix=".xml", dir=MODELS)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(room_scene_xml(layout))
        return TB3Sim(path, **sim_kwargs)
    finally:
        os.remove(path)

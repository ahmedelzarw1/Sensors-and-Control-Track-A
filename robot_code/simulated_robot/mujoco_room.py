"""MuJoCo XML for the Track A room around the patched ROBOTIS TurtleBot3 Burger (mujoco_files/robot.xml).

room_scene_xml(layout) builds the room for any Layout (robot_code/room/room_map.py): walls, obstacles, charging station and
ArUco marker. The fixed demo room is written to mujoco_files/room.xml by commands/build_mujoco_files.py; random
layouts (robot_code/room/random_room.py) are built on the fly by load_room_sim(). The map given to the robot and the simulated
world come from the same Layout, so they agree by construction.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from robot_code.room import room_map as km
from robot_code.vision.camera_maths import CAMERA_HEIGHT, CAMERA_WIDTH

MODELS = Path(__file__).resolve().parents[2] / "mujoco_files"
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
    obstacles = km.OBSTACLES if layout is None else layout.obstacles
    for ob in obstacles:
        geoms.append(_obstacle_geom(ob))
    sx = x1 - km.STATION_DEPTH / 2
    geoms.append(f'    <geom name="charging_station" type="box" pos="{sx:g} {km.DOCK_Y:g} {km.STATION_HEIGHT / 2:g}" '
                 f'size="{km.STATION_DEPTH / 2:g} {km.STATION_HALF_WIDTH:g} {km.STATION_HEIGHT / 2:g}" '
                 'material="station_mat"/>')
    plate = km.MARKER_SIZE / 2 + km.MARKER_WHITE_BORDER
    mx = km.STATION_FRONT_X - km.MARKER_PLATE_THICKNESS / 2
    # MuJoCo draws a 2d texture unmirrored on the box face whose normal is the local +z axis. Local +z is the marker
    # outward normal (world -x), local y is world +z (image up), local x is world -y (image right for a viewer).
    geoms.append(f'    <body name="aruco_marker" pos="{mx:g} {km.DOCK_Y:g} {km.MARKER_Z:g}" xyaxes="0 -1 0 0 0 1">\n'
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
    from robot_code.simulated_robot.simulator import TB3Sim
    fd, path = tempfile.mkstemp(prefix=".layout_", suffix=".xml", dir=MODELS)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(room_scene_xml(layout))
        return TB3Sim(path, **sim_kwargs)
    finally:
        os.remove(path)

"""Generate the MuJoCo scenes from the unmodified official ROBOTIS TurtleBot3 Burger model.

Source: third_party/robotis_mujoco_menagerie/robotis_tb3/ (ROBOTIS, Apache 2.0). Three files are written to models/
and none may be edited by hand:

1. models/tb3_burger.xml: the robot with the patches below, plus the camera and LiDAR frame this project adds.
2. models/track_a_scene.xml: the ROBOTIS open-floor scene with the patched robot and one test obstacle, used for the
   platform characterisation, the odometry calibration and the platform tests.
3. models/track_a_room.xml: the patched robot in the Track A room. The room, obstacles, charging station and ArUco
   marker are built from src/known_map.py, so the map the robot is given and the simulated world agree by construction.

Why the robot is patched. Driven as shipped, the model does not follow velocity commands. Commanding 0.1 m/s for 3 s
moved it 39 percent of the distance, a 1 rad/s turn in place reached 29 percent, and on arcs the body turned 41 percent
of what the wheel encoders reported. Three faults were found, each measured on its own and together in
docs/platform_notes.md (the numbers are reproduced by tests/test_platform.py):
- The chassis collision mesh is used through its convex hull, which reaches 0.1 mm below the floor, lower than the
  bottom of the caster sphere. The chassis therefore carries the rear of the robot and drags with friction 1.
- The caster is meant to be frictionless (condim 1, friction 0.0001), but MuJoCo combines contact properties by
  taking the larger value of the two geoms unless a priority is set, so the floor's friction and condim were used.
- The wheel velocity servo gain kv = 0.1 against the joint friction loss of 0.1 N m leaves a steady wheel speed error
  of 0.1 / 0.1 = 1 rad/s, so the wheels never reach the commanded speed.
Each fault masked the others: fixing any one alone changed little.

Every patch is asserted to apply the expected number of times. Run from the repository root:
    python scripts/make_track_a_scene.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src import known_map as km  # noqa: E402
from src.tb3.scene import FLOOR_BITS, room_scene_xml  # noqa: E402
from src.vision.camera_model import CAMERA_FOVY_DEG, CAMERA_HEIGHT, CAMERA_WIDTH  # noqa: E402

TB3_DIR = ROOT / "third_party/robotis_mujoco_menagerie/robotis_tb3"
SRC_ROBOT = TB3_DIR / "turtlebot3_burger.xml"
SRC_SCENE = TB3_DIR / "scene_turtlebot3_burger.xml"
MODELS = ROOT / "models"
DST_ROBOT = MODELS / "tb3_burger.xml"
DST_SCENE = MODELS / "track_a_scene.xml"
DST_ROOM = MODELS / "track_a_room.xml"
MARKER_PNG = MODELS / "textures/aruco_4x4_50_id0.png"

# Collision bitmasks. Two geoms collide if (contype_a & conaffinity_b) or (contype_b & conaffinity_a).
# Floor: contype 2, conaffinity 2 (FLOOR_BITS, src/tb3/scene.py). Tyres and caster: contype 1, conaffinity 3, so they
# touch the floor and obstacles. Chassis and LiDAR housing keep the default (1, 1): walls and obstacles, never the floor.

CAMERA_MOUNT = (0.056, 0.0, 0.11)  # m in the base frame (assumption, see the patch description)
LIDAR_FRAME = (-0.032, 0.0, 0.182)  # m, origin of the ROBOTIS LDS-01 mesh

ROBOT_PATCHES = [
    # (description, old, new, expected count)
    ("mesh directory relative to models/",
     'meshdir="assets/"', 'meshdir="../third_party/robotis_mujoco_menagerie/robotis_tb3/assets/"', 1),
    ("caster uses its own frictionless contact (priority), and touches the floor (conaffinity 3)",
     'solimp="0.95 0.99 0.001" condim="1"/>', 'solimp="0.95 0.99 0.001" condim="1" priority="1" conaffinity="3"/>', 1),
    ("tyres touch the floor (conaffinity 3); the chassis and LiDAR housing keep the default and do not",
     'mesh="left_tire" class="collision"/>', 'mesh="left_tire" class="collision" conaffinity="3"/>', 1),
    ("tyres touch the floor (right)",
     'mesh="right_tire" class="collision"/>', 'mesh="right_tire" class="collision" conaffinity="3"/>', 1),
    ("wheel velocity servo gain 3 (steady error 0.033 rad/s against the 0.1 N m friction loss) with a 1.4 N m torque "
     "limit (Dynamixel XL430 stall torque, an assumption)",
     '<velocity kv="0.1" ctrlrange="-6.67 6.67" ctrllimited="true"/>',
     '<velocity kv="3" ctrlrange="-6.67 6.67" ctrllimited="true" forcerange="-1.4 1.4" forcelimited="true"/>', 1),
    ("forward camera and LiDAR frame added (the Burger has neither in the ROBOTIS model)",
     '      <body name="wheel_left" pos="0 0.08 0.033"',
     f'      <body name="color_sensor" pos="{CAMERA_MOUNT[0]:g} {CAMERA_MOUNT[1]:g} {CAMERA_MOUNT[2]:g}">\n'
     '        <geom name="color_sensor_lens" type="cylinder" size="0.0125 0.006" euler="0 1.57079632679 0" '
     'material="black" contype="0" conaffinity="0" density="0" group="2"/>\n'
     f'        <camera name="color_camera" mode="fixed" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{CAMERA_FOVY_DEG:g}"/>\n'
     '      </body>\n'
     f'      <site name="base_scan" pos="{LIDAR_FRAME[0]:g} {LIDAR_FRAME[1]:g} {LIDAR_FRAME[2]:g}"/>\n'
     '      <body name="wheel_left" pos="0 0.08 0.033"', 1),
]

SCENE_PATCHES = [
    ("include the patched robot", '<include file="turtlebot3_burger.xml"/>', '<include file="tb3_burger.xml"/>', 1),
    ("offscreen buffer at the camera resolution",
     '<global azimuth="120" elevation="-20"/>',
     f'<global azimuth="120" elevation="-20" offwidth="{CAMERA_WIDTH}" offheight="{CAMERA_HEIGHT}"/>', 1),
    ("floor collision bits", 'type="plane" material="groundplane"/>', f'type="plane" material="groundplane" {FLOOR_BITS}/>', 1),
    ("test obstacle 1 m ahead of the origin, near face at x = 0.92 m (used by tests/test_platform.py)",
     '  </worldbody>',
     '    <geom name="obstacle_front" type="box" pos="1.0 0 0.2" size="0.08 0.25 0.2" rgba="0.85 0.25 0.25 1"/>\n'
     '  </worldbody>', 1),
]


def apply(text: str, patches) -> tuple[str, list]:
    notes = []
    for desc, old, new, count in patches:
        assert text.count(old) == count, f"patch not applicable ({text.count(old)} matches): {desc}"
        text = text.replace(old, new)
        notes.append(desc)
    return text, notes


def header(notes) -> str:
    return ("<!-- Generated by scripts/make_track_a_scene.py from third_party/robotis_mujoco_menagerie (ROBOTIS,\n"
            "     Apache 2.0). This file was modified from the original. Changes: " + "; ".join(notes) +
            ". Do not edit by hand. -->\n")


def write_marker_texture() -> None:
    """ArUco marker image with a one-cell white quiet zone."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, km.ARUCO_DICT))
    cells, px = km.MARKER_BITS + 2, 20
    img = cv2.aruco.generateImageMarker(dictionary, km.MARKER_ID, cells * px)
    img = cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)
    MARKER_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MARKER_PNG), cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    robot, notes = apply(SRC_ROBOT.read_text(), ROBOT_PATCHES)
    DST_ROBOT.write_text(header(notes) + robot)
    scene, snotes = apply(SRC_SCENE.read_text(), SCENE_PATCHES)
    DST_SCENE.write_text(header(snotes) + scene)
    write_marker_texture()
    DST_ROOM.write_text("<!-- Generated by scripts/make_track_a_scene.py: the Track A room from src/known_map.py with "
                        "the patched ROBOTIS TurtleBot3 Burger (models/tb3_burger.xml). Do not edit by hand. -->\n"
                        + room_scene_xml())
    for p in (DST_ROBOT, DST_SCENE, DST_ROOM, MARKER_PNG):
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()

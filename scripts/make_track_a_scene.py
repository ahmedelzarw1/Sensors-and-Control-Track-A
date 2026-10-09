"""Generate the MuJoCo scenes from the unmodified third-party TurtleBot3 MJCF.

Two files are written. Neither may be edited by hand.

1. models/track_a_scene.xml: the robot with three platform patches and the original surroundings (a few obstacles near
   the origin and a ring of posts). Used for the platform characterisation in docs/platform_notes.md and its tests.
2. models/track_a_room.xml: the same patched robot in the Track A room. The room, obstacles, charging station and
   ArUco marker are built from src/known_map.py, so the map the robot is given and the simulated world agree by
   construction. The original obstacles and the ring of posts are removed. The camera gets an explicit field of
   view and the offscreen buffer is enlarged to the camera resolution in src/vision/camera_model.py.

Why the platform patches: in the original file the caster has collisions disabled and the contact model is soft, which
makes the robot rest about 4 degrees nose-up, drag the rear edge of its body on the floor and ring after every change of
wheel command. Measured effect on the yaw change per 0.1 s control step is in docs/platform_notes.md. Every change is
asserted to apply the expected number of times, and nothing else is touched.

Run from the repository root:  python scripts/make_track_a_scene.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import known_map as km  # noqa: E402
from src.vision.camera_model import CAMERA_FOVY_DEG, CAMERA_HEIGHT, CAMERA_WIDTH  # noqa: E402

SRC = ROOT / "third_party/hakoniwa-mujoco-robots/models/tb3/turtlebot3_burger_world.xml"
DST = ROOT / "models/track_a_scene.xml"
DST_ROOM = ROOT / "models/track_a_room.xml"
MARKER_PNG = ROOT / "models/textures/aruco_4x4_50_id0.png"

PATCHES = [
    # (description, old, new, expected count)
    ("enable caster collision so the robot rests level on three contacts",
     ' solref="0.02 1" contype="0" conaffinity="0"/>', ' solref="0.02 1"/>', 1),
    ("stiffer wheel-ground contact",
     'type="cylinder" rgba="0.3 0.3 0.3 1" friction="2.0 0.005 0.0001"/>',
     'type="cylinder" rgba="0.3 0.3 0.3 1" friction="2.0 0.005 0.0001" solref="0.005 1"/>', 2),
    ("elliptic friction cones and no-slip iterations to suppress contact ringing",
     'integrator="implicit"/>', 'integrator="implicit" cone="elliptic" noslip_iterations="5"/>', 1),
]

ROOM_PATCHES = [
    ("near clip plane at about 1.4 cm (MuJoCo scales znear by the model extent, 7.2 m here, so the original 0.05 "
     "clipped everything closer than 0.36 m, including the marker when docked)",
     '<map znear="0.05" zfar="10"/>', '<map znear="0.002" zfar="10"/>', 1),
    ("offscreen buffer at the camera resolution",
     '<global offwidth="320" offheight="240"/>',
     f'<global offwidth="{CAMERA_WIDTH}" offheight="{CAMERA_HEIGHT}"/>', 1),
    ("explicit camera vertical field of view",
     '<camera name="color_camera" mode="fixed" pos="0 0 0" xyaxes="0 -1 0 0 0 1"/>',
     f'<camera name="color_camera" mode="fixed" pos="0 0 0" xyaxes="0 -1 0 0 0 1" fovy="{CAMERA_FOVY_DEG:g}"/>', 1),
    ("matte floor without the high-contrast reflective checker, so the floor does not dominate the camera image",
     'material="grid" friction="1.0"/>', 'material="floor_mat" friction="1.0"/>', 1),
]

# Original obstacles near the origin (3) and the ring of posts (12) are removed from the room scene.
ORIGINAL_OBSTACLE_BODY = re.compile(r'\n\s*<body name="(?:obstacle_\w+|ring_obstacle_\d+)"[^>]*>.*?</body>', re.S)
N_ORIGINAL_OBSTACLES = 15


def apply(text: str, patches) -> tuple[str, list]:
    notes = []
    for desc, old, new, count in patches:
        assert text.count(old) == count, f"patch not applicable ({text.count(old)} matches): {desc}"
        text = text.replace(old, new)
        notes.append(desc)
    return text, notes


def write_marker_texture() -> None:
    """ArUco marker image with a one-cell white quiet zone, one pixel per 20 texels for crisp edges."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, km.ARUCO_DICT))
    cells = km.MARKER_BITS + 2
    px_per_cell = 20
    img = cv2.aruco.generateImageMarker(dictionary, km.MARKER_ID, cells * px_per_cell)
    img = cv2.copyMakeBorder(img, px_per_cell, px_per_cell, px_per_cell, px_per_cell, cv2.BORDER_CONSTANT, value=255)
    MARKER_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MARKER_PNG), cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))


def room_assets() -> str:
    rel = MARKER_PNG.relative_to(DST_ROOM.parent).as_posix()
    return (f'    <texture name="aruco_tex" type="2d" file="{rel}"/>\n'
            '    <material name="aruco_mat" texture="aruco_tex" texrepeat="1 1" specular="0" shininess="0" emission="0.6"/>\n'
            '    <texture name="floor_tex" type="2d" builtin="checker" width="256" height="256" '
            'rgb1="0.62 0.60 0.56" rgb2="0.56 0.54 0.50"/>\n'
            '    <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0"/>\n'
            '    <material name="wall_mat" rgba="0.82 0.80 0.76 1"/>\n'
            '    <material name="station_mat" rgba="0.15 0.15 0.18 1"/>\n')


def room_bodies() -> str:
    t, h = km.WALL_THICKNESS, km.WALL_HEIGHT
    x0, x1, y0, y1 = km.ROOM_X_MIN, km.ROOM_X_MAX, km.ROOM_Y_MIN, km.ROOM_Y_MAX
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hx, hy = 0.5 * (x1 - x0) + t, 0.5 * (y1 - y0)
    walls = [
        ("wall_north", cx, y1 + t / 2, hx, t / 2),
        ("wall_south", cx, y0 - t / 2, hx, t / 2),
        ("wall_east", x1 + t / 2, cy, t / 2, hy),
        ("wall_west", x0 - t / 2, cy, t / 2, hy),
    ]
    lines = ['    <light name="ceiling" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>']
    for name, x, y, sx, sy in walls:
        lines.append(f'    <geom name="{name}" type="box" pos="{x:g} {y:g} {h / 2:g}" size="{sx:g} {sy:g} {h / 2:g}" '
                     f'material="wall_mat"/>')
    for ob in km.OBSTACLES:
        if isinstance(ob, km.Box):
            lines.append(f'    <geom name="{ob.name}" type="box" pos="{ob.x:g} {ob.y:g} {ob.height / 2:g}" '
                         f'size="{ob.hx:g} {ob.hy:g} {ob.height / 2:g}" rgba="0.85 0.45 0.25 1"/>')
        else:
            lines.append(f'    <geom name="{ob.name}" type="cylinder" pos="{ob.x:g} {ob.y:g} {ob.height / 2:g}" '
                         f'size="{ob.radius:g} {ob.height / 2:g}" rgba="0.25 0.55 0.85 1"/>')
    # Charging station body against the east wall, marker plate on its front face.
    sx = x1 - km.STATION_DEPTH / 2
    lines.append(f'    <geom name="charging_station" type="box" pos="{sx:g} {km.DOCK_Y:g} {km.STATION_HEIGHT / 2:g}" '
                 f'size="{km.STATION_DEPTH / 2:g} {km.STATION_HALF_WIDTH:g} {km.STATION_HEIGHT / 2:g}" '
                 f'material="station_mat"/>')
    plate_half = km.MARKER_SIZE / 2 + km.MARKER_WHITE_BORDER
    mx = km.STATION_FRONT_X - km.MARKER_PLATE_THICKNESS / 2
    # MuJoCo draws a 2d texture unmirrored on the box face whose normal is the local +z axis. The plate frame is
    # rotated so that local +z is the marker outward normal (world -x), local y is world +z (image up) and local x
    # is world -y (image right for a viewer facing the marker).
    lines.append(f'    <body name="aruco_marker" pos="{mx:g} {km.DOCK_Y:g} {km.MARKER_Z:g}" xyaxes="0 -1 0 0 0 1">\n'
                 f'      <geom name="aruco_plate" type="box" size="{plate_half:g} {plate_half:g} '
                 f'{km.MARKER_PLATE_THICKNESS / 2:g}" material="aruco_mat" contype="0" conaffinity="0"/>\n'
                 '    </body>')
    return "\n".join(lines) + "\n"


def main() -> None:
    base, notes = apply(SRC.read_text(), PATCHES)
    header = ("<!-- Generated by scripts/make_track_a_scene.py from third_party/hakoniwa-mujoco-robots (MIT).\n"
              "     Changes: " + "; ".join(notes) + ". Do not edit by hand. -->\n")
    DST.parent.mkdir(exist_ok=True)
    DST.write_text(header + base)
    print(f"wrote {DST.relative_to(ROOT)} with {len(PATCHES)} patches")

    room, room_notes = apply(base, ROOM_PATCHES)
    removed = len(ORIGINAL_OBSTACLE_BODY.findall(room))
    assert removed == N_ORIGINAL_OBSTACLES, f"expected {N_ORIGINAL_OBSTACLES} original obstacles, found {removed}"
    room = ORIGINAL_OBSTACLE_BODY.sub("", room)
    room, _ = apply(room, [("room assets", "  </asset>", room_assets() + "  </asset>", 1),
                           ("room geometry", '    <body name="base_footprint"',
                            room_bodies() + '\n    <body name="base_footprint"', 1)])
    write_marker_texture()
    room_notes += ["removed the 15 original obstacles and posts",
                   "added the room, obstacles, charging station and ArUco marker from src/known_map.py"]
    header = ("<!-- Generated by scripts/make_track_a_scene.py from third_party/hakoniwa-mujoco-robots (MIT).\n"
              "     Changes: " + "; ".join(notes + room_notes) + ". Do not edit by hand. -->\n")
    DST_ROOM.write_text(header + room)
    print(f"wrote {DST_ROOM.relative_to(ROOT)} and {MARKER_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

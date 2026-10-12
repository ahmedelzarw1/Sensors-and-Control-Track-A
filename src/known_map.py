"""The known map of the Track A room: walls, obstacles, charging station and ArUco marker.

This is CONFIGURATION, not ground truth. A real robot would be given the same information as a floor plan and a
surveyed marker position. The pipeline (planner, estimator, docking) may read it. The scene generator
(scripts/make_track_a_scene.py) builds the MuJoCo room from the same numbers, so the map and the simulated world agree
by construction. Nothing here says where the robot is.

Frames: world frame W (see docs/frames.md). Marker frame M: origin at the marker centre, x along the outward normal of
the marker face, z up, y = z cross x (to the right of a viewer who faces the marker).

Every number below is a design choice of the team (an assumption about the room), not a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ------------------------------------------------------------------ room
# Inner faces of the walls. The room is 4.0 m by 3.0 m.
ROOM_X_MIN, ROOM_X_MAX = -2.0, 2.0
ROOM_Y_MIN, ROOM_Y_MAX = -1.5, 1.5
WALL_THICKNESS = 0.05
WALL_HEIGHT = 0.35  # above the LiDAR plane (0.182 m) and the camera (0.110 m)


@dataclass(frozen=True)
class Box:
    """Box obstacle (planar footprint plus height), rotated by yaw about the vertical axis."""
    name: str
    x: float
    y: float
    hx: float  # half size along the box's own x axis
    hy: float  # half size along the box's own y axis
    height: float
    yaw: float = 0.0  # rad

    @property
    def bounding_radius(self) -> float:
        return float(np.hypot(self.hx, self.hy))


@dataclass(frozen=True)
class Cylinder:
    name: str
    x: float
    y: float
    radius: float
    height: float

    @property
    def bounding_radius(self) -> float:
        return float(self.radius)


def footprint_contains(ob, X, Y) -> np.ndarray:
    """True where the planar points (X, Y) lie inside the obstacle footprint."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if isinstance(ob, Box):
        c, s = np.cos(ob.yaw), np.sin(ob.yaw)
        u = c * (X - ob.x) + s * (Y - ob.y)
        v = -s * (X - ob.x) + c * (Y - ob.y)
        return (np.abs(u) <= ob.hx) & (np.abs(v) <= ob.hy)
    return np.hypot(X - ob.x, Y - ob.y) <= ob.radius


OBSTACLES = (
    Box("obstacle_wall_block", 0.00, 0.35, 0.15, 0.45, 0.30),
    Cylinder("obstacle_bin", -0.90, -0.60, 0.15, 0.30),
    Cylinder("obstacle_post", 0.90, -0.75, 0.12, 0.30),
    Box("obstacle_crate", -1.00, 0.95, 0.20, 0.12, 0.30),
    Box("obstacle_cabinet", 1.20, 1.15, 0.15, 0.15, 0.30),
)

# ------------------------------------------------------------------ charging station and marker
DOCK_Y = 0.40                      # station centre line on the east wall
STATION_DEPTH = 0.08               # station body protrudes this far from the wall (along -x)
STATION_HALF_WIDTH = 0.15
STATION_HEIGHT = 0.26

ARUCO_DICT = "DICT_4X4_50"
MARKER_ID = 0
MARKER_SIZE = 0.15                 # side of the black square, m (what solvePnP uses)
MARKER_BITS = 4                    # 4 x 4 data cells plus a one-cell black border, so 6 cells across
MARKER_WHITE_BORDER = MARKER_SIZE / (MARKER_BITS + 2)  # one cell of white quiet zone on each side
MARKER_PLATE_THICKNESS = 0.004
MARKER_HEIGHT = 0.11               # centre height, equal to the camera height so the marker is centred vertically

STATION_FRONT_X = ROOM_X_MAX - STATION_DEPTH
# Marker pose in W: (x, y, z, yaw). The face points into the room (normal along -x), so yaw = pi.
MARKER_POSE_W = np.array([STATION_FRONT_X - MARKER_PLATE_THICKNESS, DOCK_Y, np.pi])
MARKER_Z = MARKER_HEIGHT

# ------------------------------------------------------------------ docking geometry
# Distances are measured from the marker face to the base frame origin along the marker normal.
DOCK_STANDOFF = 0.25               # final docked distance (camera is then about 0.19 m from the marker)
PRE_DOCK_DISTANCE = 0.90           # where the planner hands over to the docking trajectory


def pose_in_front_of_marker(distance: float) -> np.ndarray:
    """Robot pose at a given distance in front of the marker, facing the marker."""
    mx, my, myaw = MARKER_POSE_W
    n = np.array([np.cos(myaw), np.sin(myaw)])     # outward normal
    p = np.array([mx, my]) + distance * n
    return np.array([p[0], p[1], float(np.arctan2(-n[1], -n[0]))])


DOCK_POSE = pose_in_front_of_marker(DOCK_STANDOFF)
PRE_DOCK_POSE = pose_in_front_of_marker(PRE_DOCK_DISTANCE)

# ------------------------------------------------------------------ robot footprint
# Radius of the circle about the base frame origin that encloses the robot's planar footprint: the largest planar
# distance of any collision mesh vertex of the ROBOTIS Burger (a rear corner of the chassis, at about (-0.095, -0.043) m
# in B) is 0.1045 m. Recomputed from the model in tests/test_platform.py.
ROBOT_RADIUS = 0.105

# ------------------------------------------------------------------ start poses
# Nominal start poses: the robot is placed by hand on a marked start pad. The pad position is known to the robot as a
# prior with the standard deviations below (assumption: hand placement within a few centimetres and degrees).
# START_POSES all have a line of sight to the marker within 3.1 m (checked by marker_line_of_sight, see
# tests/test_scene.py), so the robot can fix its pose before it drives. Each needs a path around at least one
# obstacle, and "centre" starts facing away from the marker so the localisation search turn is exercised.
START_POSES = {
    "north_west": np.array([-0.60, 1.25, -0.3]),
    "south_west": np.array([-0.70, -1.20, 0.3]),
    "south": np.array([0.20, -1.15, 1.2]),
    "north": np.array([0.40, 1.15, -0.3]),
    "centre": np.array([-0.45, -0.45, 2.5]),
}
# Start poses WITHOUT a line of sight to the marker (occluded by the central block or out of range). The robot must
# drive about 2 m on odometry and the start prior alone. Used as a separate, harder evaluation condition.
HIDDEN_START_POSES = {
    "far_west": np.array([-1.55, -0.10, 0.0]),
    "far_north_west": np.array([-1.60, 0.30, -0.4]),
}
ALL_START_POSES = {**START_POSES, **HIDDEN_START_POSES}
START_PRIOR_SIGMA = np.array([0.05, 0.05, np.deg2rad(3.0)])


# ------------------------------------------------------------------ layouts
@dataclass(frozen=True)
class Layout:
    """One arrangement of the room: the obstacles and the start pad.

    The walls, charging station and marker are the same in every layout. The obstacles are given to the robot as its
    map; the start pad is given to it as the prior on its start pose (the true start is the pad plus a random
    placement error, applied by the simulator). A layout is configuration, not ground truth.
    """
    obstacles: tuple
    start: np.ndarray = field(default_factory=lambda: START_POSES["centre"].copy())
    name: str = "default"
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        obs = []
        for ob in self.obstacles:
            d = {"name": ob.name, "x": ob.x, "y": ob.y, "height": ob.height}
            if isinstance(ob, Box):
                d.update(type="box", hx=ob.hx, hy=ob.hy, yaw=ob.yaw)
            else:
                d.update(type="cylinder", radius=ob.radius)
            obs.append(d)
        return {"name": self.name, "seed": self.seed, "start": [float(v) for v in self.start], "obstacles": obs}


def default_layout(start: str = "centre") -> Layout:
    """The fixed room used for the demo, with one of the named start pads."""
    return Layout(OBSTACLES, ALL_START_POSES[start].copy(), name=f"default/{start}")


def station_footprint() -> Box:
    return Box("charging_station", ROOM_X_MAX - STATION_DEPTH / 2, DOCK_Y, STATION_DEPTH / 2,
               STATION_HALF_WIDTH, STATION_HEIGHT)


def obstacle_footprints(layout: Optional[Layout] = None) -> list:
    """All static obstacles in the room as Box and Cylinder footprints, including the charging station."""
    obstacles = OBSTACLES if layout is None else layout.obstacles
    return list(obstacles) + [station_footprint()]


def segment_hits_obstacle(a, b, n: int = 200, layout: Optional[Layout] = None) -> bool:
    """True if the straight segment a-b (planar) crosses any obstacle footprint or the charging station body.
    All obstacles are taller than the camera, so this is also the camera line-of-sight test on the known map."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    P = np.asarray(a, float)[:2] * (1 - t) + np.asarray(b, float)[:2] * t
    return any(np.any(footprint_contains(ob, P[:, 0], P[:, 1])) for ob in obstacle_footprints(layout))


def marker_line_of_sight(xy, layout: Optional[Layout] = None) -> bool:
    """True if the marker centre is visible from xy on the known map (in front of the marker and not occluded).
    The end point is moved 1 cm in front of the marker face so the station body itself does not count."""
    mx, my, myaw = MARKER_POSE_W
    n = np.array([np.cos(myaw), np.sin(myaw)])
    front = np.array([mx, my]) + 0.01 * n
    if (np.asarray(xy, float)[:2] - front) @ n <= 0.0:
        return False
    return not segment_hits_obstacle(xy, front, layout=layout)

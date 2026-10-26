"""The Track A room: walls, obstacles, charging station and ArUco marker, and the start and end points of a run.

What the robot is told (like a floor plan handed to a real robot): the walls, its start point, and roughly where the
charging station is. What it is NOT told: the obstacles (it finds them with the LiDAR) and the exact station position
(it finds the marker with the camera at the end). The simulator builds the MuJoCo room from the same numbers.

Frames: world frame W (MuJoCo world, z up). Marker frame M: origin at the marker centre, x along the outward normal of
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
# The station stands against the east wall. Its position along the wall (dock_y) changes from room to room, and the
# robot is only told it approximately (Layout.dock_hint_y, within DOCK_HINT_ERROR). The camera finds the exact marker
# at the end of the trip.
DOCK_Y = 0.40                      # station centre line in the fixed demo room (mujoco_files/room.xml)
DOCK_HINT_ERROR = 0.20             # m, the true station is within this distance of the position given (assumption)
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
MARKER_Z = MARKER_HEIGHT
STATION_FRONT_X = ROOM_X_MAX - STATION_DEPTH


def marker_pose(dock_y: float) -> np.ndarray:
    """Marker pose in W (x, y, yaw) for a station at dock_y. The face points into the room (normal along -x)."""
    return np.array([STATION_FRONT_X - MARKER_PLATE_THICKNESS, dock_y, np.pi])


# ------------------------------------------------------------------ docking geometry
# Distances are measured from the marker face to the base frame origin along the marker normal.
DOCK_STANDOFF = 0.25               # final docked distance (camera is then about 0.19 m from the marker)
APPROACH_DISTANCE = 0.90           # the end point of the trip: where the camera takes over (visual servoing)


def pose_in_front_of_marker(distance: float, dock_y: float) -> np.ndarray:
    """Robot pose at a given distance in front of the marker of a station at dock_y, facing the marker."""
    mx, my, myaw = marker_pose(dock_y)
    n = np.array([np.cos(myaw), np.sin(myaw)])     # outward normal
    p = np.array([mx, my]) + distance * n
    return np.array([p[0], p[1], float(np.arctan2(-n[1], -n[0]))])


def dock_pose(dock_y: float) -> np.ndarray:
    return pose_in_front_of_marker(DOCK_STANDOFF, dock_y)


def approach_pose(dock_y: float) -> np.ndarray:
    return pose_in_front_of_marker(APPROACH_DISTANCE, dock_y)


# The dock pose in the marker frame M is the same in every room: DOCK_STANDOFF out along the normal, facing the marker.
DOCK_POSE_M = np.array([DOCK_STANDOFF, 0.0, np.pi])

# ------------------------------------------------------------------ robot footprint
# Radius of the circle about the base frame origin that encloses the robot's planar footprint: the largest planar
# distance of any collision mesh vertex of the ROBOTIS Burger (a rear corner of the chassis, at about (-0.095, -0.043) m
# in B) is 0.1045 m.
ROBOT_RADIUS = 0.105

# ------------------------------------------------------------------ start points
# The robot starts at the west end of the room and the charging station is at the east end. The start point is placed
# by hand, so the robot knows it as a prior with the standard deviations below (assumption: hand placement within a few
# centimetres and degrees).
START_X = ROOM_X_MIN + 0.35
START_POINTS = {
    "west_south": np.array([START_X, -1.00, 0.0]),
    "west_centre": np.array([START_X, 0.00, 0.0]),
    "west_north": np.array([START_X, 1.00, 0.0]),
}
START_PRIOR_SIGMA = np.array([0.05, 0.05, np.deg2rad(3.0)])


# ------------------------------------------------------------------ layouts
@dataclass(frozen=True)
class Layout:
    """One arrangement of the room: the obstacles, the start point and the charging station.

    Given to the robot: the start point (the prior of its EKF) and dock_hint_y (the approximate station position).
    Not given to the robot: the obstacles (the LiDAR finds them) and the true dock_y (the camera finds the marker). The
    simulator builds the room from all of it, and the evaluation measures the docking error against the true dock.
    """
    obstacles: tuple
    start: np.ndarray = field(default_factory=lambda: START_POINTS["west_centre"].copy())
    dock_y: float = DOCK_Y
    dock_hint_y: float = DOCK_Y
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
        return {"name": self.name, "seed": self.seed, "start": [float(v) for v in self.start],
                "dock_y": float(self.dock_y), "dock_hint_y": float(self.dock_hint_y), "obstacles": obs}


def default_layout(start: str = "west_centre") -> Layout:
    """The fixed demo room, with one of the start points. The robot is told the station is 0.15 m from where it is."""
    return Layout(OBSTACLES, START_POINTS[start].copy(), dock_y=DOCK_Y, dock_hint_y=DOCK_Y - 0.15,
                  name=f"default/{start}")


def station_footprint(dock_y: float = DOCK_Y) -> Box:
    return Box("charging_station", ROOM_X_MAX - STATION_DEPTH / 2, dock_y, STATION_DEPTH / 2,
               STATION_HALF_WIDTH, STATION_HEIGHT)


def obstacle_footprints(layout: Optional[Layout] = None) -> list:
    """All static obstacles in the room as Box and Cylinder footprints, including the charging station."""
    layout = layout if layout is not None else default_layout()
    return list(layout.obstacles) + [station_footprint(layout.dock_y)]

"""Plot helpers shared by the demo, the trials and the documentation figures."""
from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np

from robot_code.room import room_map as km


def draw_map(ax, show_dock: bool = True, layout=None) -> None:
    """Room walls, obstacles (of the given layout, default the fixed room), charging station, marker and dock pose."""
    ax.add_patch(mpatches.Rectangle((km.ROOM_X_MIN, km.ROOM_Y_MIN), km.ROOM_X_MAX - km.ROOM_X_MIN,
                                    km.ROOM_Y_MAX - km.ROOM_Y_MIN, fill=False, lw=2.0, ec="0.2"))
    for ob in km.obstacle_footprints(layout):
        if isinstance(ob, km.Box):
            fc = "0.25" if ob.name == "charging_station" else "#d9875a"
            c, s = np.cos(ob.yaw), np.sin(ob.yaw)
            corners = np.array([[-ob.hx, -ob.hy], [ob.hx, -ob.hy], [ob.hx, ob.hy], [-ob.hx, ob.hy]])
            pts = corners @ np.array([[c, s], [-s, c]]) + [ob.x, ob.y]
            ax.add_patch(mpatches.Polygon(pts, closed=True, fc=fc, ec="none"))
        else:
            ax.add_patch(mpatches.Circle((ob.x, ob.y), ob.radius, fc="#5a8cd9", ec="none"))
    mx, my, myaw = km.MARKER_POSE_W
    half = km.MARKER_SIZE / 2
    ax.plot([mx - half * np.sin(myaw), mx + half * np.sin(myaw)], [my + half * np.cos(myaw), my - half * np.cos(myaw)],
            color="k", lw=4, solid_capstyle="butt")
    if show_dock:
        for pose, lab in [(km.DOCK_POSE, "dock"), (km.PRE_DOCK_POSE, "pre-dock")]:
            ax.plot(pose[0], pose[1], "k+", ms=8)
            ax.annotate(lab, (pose[0], pose[1]), textcoords="offset points", xytext=(-10, 8), fontsize=7)
    ax.set_xlim(km.ROOM_X_MIN - 0.1, km.ROOM_X_MAX + 0.1)
    ax.set_ylim(km.ROOM_Y_MIN - 0.1, km.ROOM_Y_MAX + 0.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def robot_triangle(ax, pose, color, size: float = 0.12, **kw):
    x, y, th = pose
    pts = np.array([[size, 0.0], [-0.6 * size, 0.5 * size], [-0.6 * size, -0.5 * size]])
    c, s = np.cos(th), np.sin(th)
    P = pts @ np.array([[c, s], [-s, c]]) + [x, y]
    return ax.add_patch(mpatches.Polygon(P, closed=True, fc=color, ec="k", lw=0.5, **kw))


def ellipse_points(mean_xy, P_xy, n_sigma: float = 3.0, n: int = 60) -> np.ndarray:
    """Points of the n-sigma covariance ellipse of a 2D position."""
    w, V = np.linalg.eigh(0.5 * (P_xy + P_xy.T))
    w = np.maximum(w, 0.0)
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    return (V @ (np.sqrt(w)[:, None] * circ) * n_sigma).T + np.asarray(mean_xy)

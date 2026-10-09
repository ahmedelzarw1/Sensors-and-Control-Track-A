"""Occupancy grid, clearance map and A* search.

The same grid type is filled in two ways:
- from the known map (src/known_map.py), the Stage 1 planner and the fallback, and
- from LiDAR scans placed with the ESTIMATED pose (src/planning/lidar_map.py), the Stage 2 planner.

Obstacle inflation uses the Euclidean distance transform: every cell stores its distance to the nearest occupied cell
(the clearance). A cell is traversable if its clearance is at least the robot radius plus a safety margin, which is
the same as inflating every obstacle by that amount. A* then searches the traversable cells (8-connected) with a small
extra cost near obstacles so the path keeps away from them when there is room.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from src import known_map as km


@dataclass
class Grid:
    x0: float          # world x of the centre of cell (0, 0)
    y0: float
    res: float         # cell size, m
    occ: np.ndarray    # bool (ny, nx), True = occupied

    @property
    def shape(self):
        return self.occ.shape

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((y - self.y0) / self.res)), int(round((x - self.x0) / self.res))

    def to_world(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + j * self.res, self.y0 + i * self.res

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        ii, jj = np.mgrid[0:self.occ.shape[0], 0:self.occ.shape[1]]
        return self.x0 + jj * self.res, self.y0 + ii * self.res

    def clearance(self) -> np.ndarray:
        """Distance from each cell centre to the nearest occupied cell centre, m."""
        return distance_transform_edt(~self.occ) * self.res


def empty_room_grid(res: float = 0.05, pad: float = 0.0) -> Grid:
    """Grid covering the room interior. Cells outside the room count as occupied (the walls)."""
    x0, y0 = km.ROOM_X_MIN - pad, km.ROOM_Y_MIN - pad
    nx = int(round((km.ROOM_X_MAX - km.ROOM_X_MIN + 2 * pad) / res)) + 1
    ny = int(round((km.ROOM_Y_MAX - km.ROOM_Y_MIN + 2 * pad) / res)) + 1
    g = Grid(x0, y0, res, np.zeros((ny, nx), bool))
    X, Y = g.centres()
    g.occ |= (X <= km.ROOM_X_MIN) | (X >= km.ROOM_X_MAX) | (Y <= km.ROOM_Y_MIN) | (Y >= km.ROOM_Y_MAX)
    return g


def known_map_grid(res: float = 0.05) -> Grid:
    """Rasterise the known map: walls, obstacles and the charging station."""
    g = empty_room_grid(res)
    X, Y = g.centres()
    for ob in km.obstacle_footprints():
        if isinstance(ob, km.Box):
            g.occ |= (np.abs(X - ob.x) <= ob.hx) & (np.abs(Y - ob.y) <= ob.hy)
        else:
            g.occ |= np.hypot(X - ob.x, Y - ob.y) <= ob.radius
    return g


_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def astar(grid: Grid, start_xy, goal_xy, clearance_min: float, clearance_pref: float = 0.35,
          near_cost: float = 2.0) -> list[tuple[float, float]] | None:
    """Shortest 8-connected path between two world points over cells with clearance >= clearance_min.

    Cost per step is the step length times (1 + near_cost * max(0, clearance_pref - c) / clearance_pref), so cells
    closer than clearance_pref to an obstacle are discouraged but allowed. Returns world points or None.
    """
    clr = grid.clearance()
    free = clr >= clearance_min
    s, g = grid.to_cell(*start_xy), grid.to_cell(*goal_xy)
    # The start may lie inside the inflated zone (the robot itself is never inside an obstacle). Allow leaving it.
    if not grid.inside(*s) or not grid.inside(*g) or not free[g]:
        return None
    penalty = 1.0 + near_cost * np.clip(clearance_pref - clr, 0.0, None) / clearance_pref
    h = lambda c: grid.res * np.hypot(c[0] - g[0], c[1] - g[1])
    open_ = [(h(s), 0.0, s)]
    came, cost = {s: None}, {s: 0.0}
    while open_:
        _, gc, cur = heapq.heappop(open_)
        if cur == g:
            break
        if gc > cost[cur]:
            continue
        for di, dj in _NEIGHBOURS:
            nb = (cur[0] + di, cur[1] + dj)
            if not grid.inside(*nb):
                continue
            if not free[nb] and not (not free[cur] and clr[nb] >= clr[cur]):
                continue  # blocked, unless we are still escaping the inflated zone around the start
            step = grid.res * (1.4142135623730951 if di and dj else 1.0) * penalty[nb]
            nc = gc + step
            if nc < cost.get(nb, np.inf):
                cost[nb], came[nb] = nc, cur
                heapq.heappush(open_, (nc + h(nb), nc, nb))
    if g not in came:
        return None
    cells, c = [], g
    while c is not None:
        cells.append(c)
        c = came[c]
    cells.reverse()
    pts = [grid.to_world(*c) for c in cells]
    pts[0], pts[-1] = tuple(start_xy), tuple(goal_xy)
    return pts


def segment_clear(clr: np.ndarray, grid: Grid, a, b, clearance_min: float) -> bool:
    """True if every grid sample along the segment a-b has clearance >= clearance_min."""
    n = max(2, int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * grid.res))) + 1)
    for t in np.linspace(0.0, 1.0, n):
        i, j = grid.to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if not grid.inside(i, j) or clr[i, j] < clearance_min:
            return False
    return True

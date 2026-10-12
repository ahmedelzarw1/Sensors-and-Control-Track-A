"""Path planning: a path around the obstacles that the robot can actually drive.

astar_grid.py        occupancy grid of the map, obstacle inflation, A* search
smooth_path.py        shortcutting, circular-arc corners, speed profile, time sampling
plan_path.py     plan_reference: start pose -> goal pose as a ReferenceTrajectory
reference_path.py   integrate_reference: builds the reference with the unicycle model so it is drivable
"""

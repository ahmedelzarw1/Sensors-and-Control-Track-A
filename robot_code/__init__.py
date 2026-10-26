"""Everything that runs ON the robot, one file per topic. It never sees the true position (only
experiments/true_position.py reads it, for the evaluation).

    robot_brain.py       decides what to do every 0.1 s: localise, plan, drive, then find the marker and dock
    unicycle_model.py    how the robot moves: model, linearisation, controllability, observability
    lqr_controller.py    LQR tracker and command smoothing
    kalman_filter.py     EKF: odometry + LiDAR fix (whole trip), odometry + camera (docking) -> "where am I?"
    vision.py            camera maths, ArUco marker detection, measurement noise
    lidar.py             LiDAR position fix from the walls, occupancy map of the obstacles, safety stop
    path_planning.py     A*, smooth path, speed profile
    docking.py           visual servoing into the station (marker frame), stop rule, marker search
    room_map.py          the room: walls, station and marker, start and end points
    random_room.py       random rooms: start at the west end, station on the east wall, no-clash rules
    simulator.py         MuJoCo: the robot, wheels, odometry, LiDAR scan, camera, building the room
    data_types.py        the messages the parts pass to each other
"""

"""Everything that runs ON the robot, one file per topic. It never sees the true position
(tests/test_robot_never_sees_truth.py).

    robot_brain.py       decides what to do every 0.1 s: find the marker, plan, drive, dock
    unicycle_model.py    how the robot moves: model, linearisation, controllability, observability
    lqr_controller.py    LQR tracker and command smoothing
    kalman_filter.py     EKF: odometry + marker -> "where am I?"
    vision.py            camera maths, ArUco marker detection, measurement noise
    lidar.py             LiDAR occupancy map of the obstacles and the LiDAR safety stop
    path_planning.py     A*, smooth path, speed profile
    docking.py           final approach, stop rule, marker search
    room_map.py          the room the robot is given (walls, dock, marker, start pads)
    random_room.py       random rooms with the no-clash rules
    simulator.py         MuJoCo: the robot, wheels, odometry, LiDAR scan, camera, building the room
    data_types.py        the messages the parts pass to each other
"""

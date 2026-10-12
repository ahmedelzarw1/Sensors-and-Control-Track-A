"""Everything that runs ON the robot. It never sees the true position (tests/test_robot_never_sees_truth.py).

Folders, in the order the data flows:
    room/               the room the robot is given as its map, and random rooms
    simulated_robot/    the TurtleBot3 in MuJoCo: wheels and odometry, LiDAR, camera, the MuJoCo room
    vision/             camera image -> ArUco marker -> measurement
    localisation/       Kalman filter: odometry + marker -> "where am I?"
    path_planning/      A* path around the obstacles -> smooth reference path
    controller/         LQR: where I am + where I should be -> speed and turn commands
    docking/            final approach, stop rule, marker search
    movement_maths/     the maths of movement used by localisation, path planning and the controller
    robot_brain.py      decides what to do every 0.1 s, using all of the above
    data_types.py       the messages the parts pass to each other
"""

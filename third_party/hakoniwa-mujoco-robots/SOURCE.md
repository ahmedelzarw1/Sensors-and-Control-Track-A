# Third-party source

TurtleBot3 Burger MuJoCo model and LDS-01 LiDAR specification, copied unmodified from:

- Repository: https://github.com/hakoniwalab/hakoniwa-mujoco-robots
- Commit: 77b036edff052abf1877940848acd513bea11a2f
- Licence: MIT (see LICENSE in this folder)
- Files: models/tb3/turtlebot3_burger_world.xml, config/sensors/lidar/lds-01.json

Notes
- The MJCF defines the robot, two velocity actuators and a forward camera. It does not define a LiDAR sensor, an ArUco marker or a charging station. Those are added by this project.
- Any modifications must be made in a copy outside this folder, and listed in the project README.

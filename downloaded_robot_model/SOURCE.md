# Third-party source

Official ROBOTIS MuJoCo model of the TurtleBot3 Burger, copied unmodified from:

- Repository: https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie
- Commit: d8344c0dbe7a00208d0301111523dde65efc174a (2 February 2026)
- Licence: Apache License 2.0 (see robotis_tb3/LICENSE)
- Files: robotis_tb3/turtlebot3_burger.xml, robotis_tb3/scene_turtlebot3_burger.xml, and the four meshes the Burger
  uses in robotis_tb3/assets/ (burger_base.stl, lds.stl, left_tire.stl, right_tire.stl). The Waffle Pi files and
  images are not included.

Notes
- The model defines the robot, its meshes, mass properties, a rear caster and two velocity actuators. It defines no
  camera and no LiDAR sensor (the real Burger has no camera). Those are added by this project.
- These files are not edited. commands/build_mujoco_files.py writes patched copies to mujoco_files/ and lists every change
  (Apache 2.0 section 4(b): modified files carry a notice stating that they were changed).

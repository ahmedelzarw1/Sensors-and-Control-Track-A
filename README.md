# 41014 Integrated Project, Track A: visual docking of a TurtleBot3 in MuJoCo

The official ROBOTIS TurtleBot3 Burger model in MuJoCo localises itself from an ArUco marker on a charging station, plans a path around the
obstacles in a room, tracks it with an LQR controller acting on an EKF estimate, and docks at the station. The
controller never receives ground truth. Ground truth is read only by `experiments/` to measure the results.

Status: Stage 1 (end-to-end simulation) working. LiDAR-based navigation (Stage 2) and the Monte Carlo evaluation
(Stage 3) are in progress.

![Path of the north_west demo run](results/demo/north_west/path.png)

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Tested on macOS (Apple silicon) with Python 3.12 and MuJoCo 3.15.0, using MuJoCo's default rendering backend. On Linux
without a display, set `MUJOCO_GL=egl` (or `osmesa`) before running. The Linux backends have not been tested.

## Running the simulation

```
python commands/run_demo.py                 # all five start pads; video of the first
python commands/run_demo.py --video-all     # a video for every start pad
python commands/run_demo.py --hidden        # also the two start pads without a view of the marker
```

Output goes to `results/demo/<start>/`: `demo.mp4` (top-down view with the true path, the EKF estimate and its 3 sigma
ellipse, next to the robot camera view), `log.csv`, and plots of the path, the estimation errors with 3 sigma
bounds, NEES and NIS, and the control effort. `results/demo/summary.csv` collects the runs.

To watch a run live in the interactive MuJoCo viewer (add `--random` for a random layout, which also prints the
serial monitor):

```
python .venv/bin/mjpython commands/watch_live.py --start centre     # macOS
python .venv/bin/mjpython commands/watch_live.py --random 3         # macOS, random layout seed 3
python commands/watch_live.py --start centre                        # Linux
```

Random layouts (random obstacles and start pad, checked for clashes; see below):

```
python commands/run_random_rooms.py                     # one new random layout, with the serial monitor printout
python commands/run_random_rooms.py --seed 3            # reproduce layout 3
python commands/run_random_rooms.py --seed 100 --count 30 --quiet --no-video   # 30 layouts and a success rate
```

## Robot model

The robot is the official ROBOTIS TurtleBot3 Burger MuJoCo model (`downloaded_robot_model/`, Apache 2.0),
used unmodified. `commands/build_mujoco_files.py` writes a patched copy to `mujoco_files/robot.xml` and builds the
scenes around it. Every patch is asserted to apply and is listed in the generated file.

Driven as shipped, the model did not follow velocity commands. Three faults were found, and each one masked the
others, so fixing any single one changed little:

1. The chassis collision mesh (its convex hull) reaches below the caster, so the chassis carried the rear of the robot
   and dragged on the floor. Collision bitmasks now keep the chassis off the floor, while it still hits walls and
   obstacles.
2. The caster is meant to be frictionless, but MuJoCo uses the larger friction of two touching surfaces unless a
   priority is set. The caster now has priority, so its own frictionless contact is used.
3. The wheel servo gain (kv = 0.1) against the joint friction (0.1 N m) left a steady wheel speed error of 1 rad/s.
   The gain is now 3 (steady error 0.033 rad/s), with a 1.4 N m torque limit (the Dynamixel XL430 stall torque, an
   assumption).

| Test (0.1 s command period) | As shipped | Patched |
|---|---|---|
| 0.1 m/s for 3 s: distance moved / commanded | 0.39 | 0.99 |
| 1 rad/s turn in place: rotation / commanded | 0.29 | 0.98 |
| Arc at 0.15 m/s, 0.5 rad/s: true heading change / encoder heading change | 0.41 | 0.99 |
| Slide after a stop from 0.15 m/s | | 0.4 mm |

The ROBOTIS Burger has no camera and no LiDAR sensor. This project adds a forward camera 0.056 m ahead of and 0.11 m
above the base origin (an assumption: the real Burger ships without a camera), and casts the LiDAR beams from the
origin of the ROBOTIS LDS-01 mesh using the LDS-01 specification. The ROBOTIS wheel limit of 6.67 rad/s gives the
tracker limits of 0.22 m/s and 2.75 rad/s. `tests/test_simulated_robot.py` checks the geometry, the contacts, the drive
behaviour and the robot radius against the model.

## Random layouts

`robot_code/room/random_room.py` draws a new room arrangement from a seed: 3 to 5 obstacles (boxes of any rotation and
cylinders, random sizes and positions) and a random start pad and heading. Every layout is reproducible from its seed.
Candidates are redrawn until every rule holds:

- Obstacles are at least 0.40 m apart and at least 0.40 m from the walls, measured between the circles that enclose
  them. The robot needs 2 x (0.105 m radius + 0.075 m margin) = 0.36 m to pass, so every gap is passable and nothing
  touches.
- No obstacle enters the docking corridor in front of the charging station.
- The start pad is at least 0.255 m from every obstacle and 0.305 m from every wall, outside the docking corridor and at
  least 1 m from the pre-dock pose. By default the marker must be in view from it (`--allow-hidden` drops this rule).
- The planner must find a path, otherwise the layout is redrawn.
- After the MuJoCo world is built and the robot is placed (the start pad plus a random placement error), MuJoCo must
  report no contact between the robot and any obstacle or wall.

`tests/test_random_rooms.py` checks every rule on 40 seeds, that the exact obstacle shapes never overlap, that the
MuJoCo world matches the layout, and that MuJoCo reports no robot contact on 10 seeds. The robot is given the layout as
its map, and the start pad as the prior on its start pose.

Outputs in `results/random_rooms/seed_<n>/` (example in the repository: `seed_3`):

| File | What it shows |
|---|---|
| `layout.png` | the layout with the clash rules drawn: bounding circles, the gap rule, the docking corridor, the planned path |
| `photos/`, `contact_sheet.png` | robot camera photos (marker outline, pose axes, range) next to the room view at the start, the first marker detection, every phase change and the end |
| `demo.mp4` | the whole run: top-down view next to the robot camera |
| `monitor.txt` | the serial-monitor printout (below) |
| `layout.json`, `log.csv`, plots | the layout and the checks it passed, the full log, the plots of `run_demo.py` |

The terminal works like a serial monitor: first the randomised layout and the checks it passed, then twice a second
what the robot calculates. Columns: EKF estimate and its 1 sigma uncertainty, LQR command, marker range and bearing,
NIS and whether the filter used the measurement, estimated lateral tracking error, and (evaluation only, never seen by
the robot) the true estimation error.

```
  t[s] phase      x^[m]  y^[m] th^[deg] sd_xy[mm] sd_th[deg]  v[m/s] w[rad/s] marker r[m] b[deg]    NIS used e_y[mm] true err[mm]
   0.0 localise   0.139 -0.897     -0.6      70.7       3.00   0.000    0.300           -      -      -            -        163.5
   0.5 >> localised from 3 marker fixes
   0.5 track      0.158 -0.908      5.0      46.6       1.21   0.041    0.119        2.19   31.6   0.13  yes     0.0        143.7
   1.0 track      0.188 -0.908     10.3      45.1       1.20   0.106    0.365        2.19   26.7   2.46  yes     4.7        137.3
```

## Where everything is

```
robot_code/                  EVERYTHING THAT RUNS ON THE ROBOT (it never sees the true position)
  robot_brain.py             decides what to do every 0.1 s: find marker, plan, drive, dock
  data_types.py              the messages the parts pass to each other
  room/                      the map: room_map.py (the room the robot is given), random_room.py (random rooms)
  simulated_robot/           the TurtleBot3 in MuJoCo: simulator.py (wheels, odometry, LiDAR), camera.py,
                             mujoco_room.py (builds the room)
  movement_maths/            unicycle_model.py: how the robot moves, linearisation, controllability
  vision/                    marker_detection.py (find the ArUco marker), camera_maths.py, marker_noise.py
  localisation/              kalman_filter.py: "where am I?" (the EKF)
  path_planning/             astar_grid.py (A*), smooth_path.py, plan_path.py, reference_path.py
  controller/                lqr_controller.py (LQR), smooth_commands.py
  docking/                   docking.py: final approach and stop

experiments/                 RUNS THE ROBOT AND CHECKS IT AGAINST THE TRUTH
  run_simulation.py          one full run in MuJoCo
  true_position.py           the only place the true position is read
  scores.py                  errors, NEES, success-rate intervals
  calibration/               measures the numbers in settings/
  outputs/                   plots, video, photos, serial monitor

commands/                    THE THINGS YOU RUN: run_demo.py, run_random_rooms.py, watch_live.py, build_mujoco_files.py
settings/                    measured numbers the robot reads (camera noise, wheel calibration, filter tuning, LiDAR)
mujoco_files/                robot.xml, room.xml, open_floor.xml (built by commands/build_mujoco_files.py)
downloaded_robot_model/      the untouched official ROBOTIS TurtleBot3 Burger model
tests/                       automatic checks: python -m pytest -q
results/                     demo/ and random_rooms/
```

Each folder's `__init__.py` lists what is inside it.

| Question | File and function |
|---|---|
| How does the robot move (kinematics)? | `robot_code/movement_maths/unicycle_model.py`: `f_unicycle` |
| How are (v, omega) turned into wheel speeds, and wheel encoders into odometry? | `robot_code/simulated_robot/simulator.py`: `wheel_speeds`, `step` |
| Where is the physics simulated? | `robot_code/simulated_robot/simulator.py`: `step` (calls `mujoco.mj_step` 50 times per 0.1 s) |
| State-space model, linearisation, discretisation | `robot_code/movement_maths/unicycle_model.py`: `tracking_error`, `linearise`, `zoh` |
| Controllability and observability | `robot_code/movement_maths/unicycle_model.py`: `ctrb`, `observability_matrix_ltv`, `H_marker_pose`; checks in `tests/test_movement_maths.py` |
| LQR (Q, R, gain) | `robot_code/controller/lqr_controller.py`: `bryson`, `dlqr`, `design_gain`, `LQRTracker.command` |
| Open loop vs closed loop | `robot_code/controller/lqr_controller.py`: `LQRTracker.command` (reference speeds plus feedback on the estimate) |
| EKF predict and update | `robot_code/localisation/kalman_filter.py`: `MarkerEKF.predict`, `MarkerEKF.update_marker` |
| Where do Q and R come from? | `experiments/calibration/` and the values in `settings/` |
| Camera model and frames | `robot_code/vision/camera_maths.py`, `robot_code/vision/marker_detection.py`: `to_base_frame` |
| Marker detection and pose | `robot_code/vision/marker_detection.py`: `ArucoDetector.detect`, `MarkerSensor.measure` |
| Path around the obstacles | `robot_code/path_planning/astar_grid.py`: `astar`; `robot_code/path_planning/smooth_path.py`: `fillet_path`, `speed_profile` |
| Docking and the stop rule | `robot_code/docking/docking.py`: `docking_reference`, `should_stop` |
| What does the robot do in each phase? | `robot_code/robot_brain.py`: `DockingMission.step`, `_decide` |
| Random layouts and the no-clash rules | `robot_code/room/random_room.py`: `sample_layout` |
| How is ground truth kept away from the robot? | `experiments/true_position.py`; enforced by `tests/test_robot_never_sees_truth.py` |

## How it works

1. **Localise.** The robot turns towards where its start prior says the marker is, and collects three accepted marker
   fixes before it drives.
2. **Plan.** A* on an occupancy grid of the known map, with obstacles inflated by the robot radius plus a margin,
   then line-of-sight shortcutting, circular-arc corners and a speed profile, giving a time-indexed reference.
3. **Track.** LQR on the linearised tracking-error model, with gains rescheduled along the reference, acting on the
   EKF estimate.
4. **Estimate.** The EKF predicts with calibrated wheel odometry and updates with the marker pose in the robot frame
   (position and yaw), which makes the full pose observable from one marker. A chi-square gate rejects outliers,
   including the two-fold pose ambiguity of a square marker.
5. **Dock.** A smooth curve onto the marker normal, then a slow straight approach. The robot stops when the
   estimated distance to the dock reaches zero, and docking is abandoned for a new marker search if no marker fix
   arrives for 3 s.

Values that come from measurement are produced by scripts and stored in `settings/`:

| Value | Script | Result |
|---|---|---|
| Marker range bias and noise model | `experiments/calibration/calibrate_camera.py` | range bias corrected (residual mean -0.4 mm on an independent sweep), detection 94 to 99 percent at every range up to 3.6 m when the marker is fully in view |
| In-motion vision noise | `experiments/calibration/camera_noise_while_driving.py` | yaw noise 2.25 times the static model, position within the static model |
| Odometry calibration | `experiments/calibration/calibrate_wheels.py` | v scale 0.997, omega scale 0.992 (effective wheel separation 0.161 m) |
| EKF consistency tuning | `experiments/calibration/tune_kalman_filter.py` | process noise x5, yaw measurement std x2: mean NEES 2.93, 0.05 percent of updates gated out (12 settings x 10 runs, all docked) |

## Results

One run per start pad, seed 1, from `python commands/run_demo.py --hidden` (`results/demo/summary.csv`). Docking
success requires the true final pose within 3 cm along the approach, 2 cm laterally and 5 degrees (these tolerances
are our assumption for a charging dock with spring contacts).

| Start pad | Marker visible at start | Result | Error along | Error lateral | Heading error | Time |
|---|---|---|---|---|---|---|
| north_west | yes | docked | -0.1 mm | 3.0 mm | -0.45 deg | 22.4 s |
| south_west | yes | docked | -2.5 mm | -9.6 mm | 0.90 deg | 25.8 s |
| south | yes | docked | -2.1 mm | -7.9 mm | 0.90 deg | 21.9 s |
| north | yes | docked | -3.0 mm | 5.7 mm | -0.64 deg | 16.8 s |
| centre | yes | docked | -0.7 mm | -4.9 mm | 0.60 deg | 26.2 s |
| far_west | no | docked | -0.9 mm | -4.3 mm | 0.39 deg | 49.2 s |
| far_north_west | no | docked | -3.8 mm | -10.8 mm | 1.13 deg | 50.8 s |

All seven runs docked.

Random layouts, one run each (`results/random_rooms/summary_*.csv`):

| Batch | Layouts | Docked | 95 percent Wilson interval | Docking position error (median, max) |
|---|---|---|---|---|
| marker in view at the start (seeds 100 to 129) | 30 | 30 | 89 to 100 percent | 5.4 mm, 15.5 mm |
| hidden starts allowed (seeds 200 to 229) | 30 (6 hidden) | 30 (6 of 6 hidden) | 89 to 100 percent (61 to 100 percent for the 6 hidden) | 6.1 mm, 12.4 mm |

Repeated trials under varied sensor and platform conditions (noise, wheel radius mismatch, friction, pushes, marker
dropout) are the next step.

## Known limitations

- **Start pads without a view of the marker are a risk.** Both hidden pads docked in this run, but from far_west the
  estimate drifted up to 140 mm while the marker was hidden. Over hidden stretches of 0.5 to 1 m, only 67 percent of
  the position errors were inside the filter's 3 sigma bound, so the filter is overconfident when it is blind for long.
  The LiDAR is scanned but not yet used for navigation or collision detection (Stage 2).
- **Varied conditions are not yet tested.** The random layouts vary the room and the start (60 runs), but not the
  sensor noise, wheel radius, friction, pushes or marker dropout.
- **Lateral docking error uses most of its tolerance in some layouts.** Over the 60 random layouts the largest lateral
  errors were 10 to 15 mm with about 1 degree of heading error (tolerance 20 mm): the short final straight does not
  give the controller time to remove a sideways offset completely.
- **The filter is briefly overconfident after localising from far away.** In the seed 3 example the true error
  (144 mm) slightly exceeded the combined 3 sigma bound (140 mm) just after the robot localised from 2.2 m: at that
  range the marker's yaw is weak, so the rotation of the robot about the marker is poorly observed until it is closer.
- **Only 6 hidden starts were sampled.** All docked, but 6 runs are too few to claim reliability (interval 61 to 100
  percent).
- **The filter is not equally consistent in every run.** Mean NEES ranges from 1.4 to 6.4 across the seven runs (3 is
  ideal); the north_west, south_west and far_west runs are above 4.
- **The position part of the measurement noise is conservative.** Median NIS is far below its expected value, because
  of a 5 mm floor on the position standard deviation.
- **Model patches and assumptions.** The three model patches above change the ROBOTIS model; the servo gain and the
  torque limit are our choices. The camera position, the odometry noise and the docking tolerances are assumptions.

## Contribution statement

| Member | Contribution |
|---|---|
| Ahmed El Zarw | State-space model, controllability and observability analysis, LQR tracker, frames and interfaces, platform wrapper, obstacle-aware path, evaluation harness, integration |
| TODO | EKF fusing wheel odometry with camera marker measurements (baseline in `robot_code/localisation/`, to be reviewed and extended) |
| TODO | ArUco detection, camera calibration, docking trajectory (baseline in `robot_code/vision/` and `robot_code/docking/`, to be reviewed and extended) |

## Acknowledgements

- TurtleBot3 Burger MuJoCo model and meshes: ROBOTIS, `robotis_mujoco_menagerie`, Apache License 2.0
  (`downloaded_robot_model/`). The generated `mujoco_files/robot.xml` is a modified copy.
- LDS-01 LiDAR specification: ROBOTIS TurtleBot3 e-Manual (values in `settings/lidar_spec.json`).
- Linearisation, discretisation, LQR and EKF structure adapted from the weekly tutorials and from Ahmed El Zarw's
  41014 Assessment 3 work.
- The tracking-error model is the standard Kanayama form.

## Generative AI declaration

Tool: Claude (Anthropic), used through the Claude chat interface and Claude Code.

Purpose: Ahmed El Zarw specified the requirements, the design constraints and the order of work, and directed the
development. Claude Code generated most of the code and documentation in this repository under that direction. This
includes the platform wrapper, the models and the LQR tracker; the room scene, the camera rendering and the ArUco
measurement pipeline with its characterisation; the baseline EKF, docking and planning modules; the odometry
calibration and filter tuning; the mission logic; and the evaluation, demo and viewer commands. Claude Code also ran the
diagnostics that found the platform issues listed above, including the three faults in the ROBOTIS model, and adapted
the project to that model.

Each team member is responsible for reviewing and understanding the material credited to them.
TODO before submission: each teammate adds their own tool use and purpose here, and the team records who reviewed which file.

# 41014 Integrated Project, Track A: visual docking of a TurtleBot3 in MuJoCo

The official ROBOTIS TurtleBot3 Burger model in MuJoCo localises itself from an ArUco marker on a charging station, plans a path around the
obstacles in a room, tracks it with an LQR controller acting on an EKF estimate, and docks at the station. The
controller never receives ground truth. Ground truth is read only by `evaluation/` to measure the results.

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
python scripts/run_demo.py                 # all five start pads; video of the first
python scripts/run_demo.py --video-all     # a video for every start pad
python scripts/run_demo.py --hidden        # also the two start pads without a view of the marker
```

Output goes to `results/demo/<start>/`: `demo.mp4` (top-down view with the true path, the EKF estimate and its 3 sigma
ellipse, next to the robot camera view), `log.csv`, and plots of the path, the estimation errors with 3 sigma
bounds, NEES and NIS, and the control effort. `results/demo/summary.csv` collects the runs.

To watch a run live in the interactive MuJoCo viewer:

```
python .venv/bin/mjpython scripts/view_live.py --start centre     # macOS
python scripts/view_live.py --start centre                        # Linux
```

## Robot model

The robot is the official ROBOTIS TurtleBot3 Burger MuJoCo model (`third_party/robotis_mujoco_menagerie/`, Apache 2.0),
used unmodified. `scripts/make_track_a_scene.py` writes a patched copy to `models/tb3_burger.xml` and builds the
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
tracker limits of 0.22 m/s and 2.75 rad/s. `tests/test_platform.py` checks the geometry, the contacts, the drive
behaviour and the robot radius against the model.

## Repository layout

| Path | Contents |
|---|---|
| `src/known_map.py` | the room, obstacles, charging station and marker pose given to the robot (configuration, not ground truth) |
| `src/tb3/` | shared data types, the MuJoCo platform wrapper (odometry, LiDAR), the simulated camera |
| `src/model/` | unicycle and tracking-error models, Jacobians, controllability and observability helpers |
| `src/control/` | LQR tracker (fixed, scheduled, time-varying gains) and the command slew-rate limiter |
| `src/vision/` | camera intrinsics, ArUco detection, pose by solvePnP, measurement noise model |
| `src/estimation/` | EKF: odometry prediction, marker pose update, chi-square gating |
| `src/planning/` | occupancy grid, A*, path smoothing, velocity profile, reference trajectories |
| `src/docking/` | docking trajectory, stop rule, marker search |
| `src/mission.py` | the on-robot pipeline: localise, plan, track, dock |
| `evaluation/` | the only code that reads ground truth: episode runner, calibration and characterisation scripts, metrics, plots, video |
| `scripts/` | scene generation, demo runner, live viewer |
| `config/` | calibration and tuning results read by the pipeline, each written by a script in `evaluation/` |
| `models/` | generated MuJoCo scenes and the marker texture (regenerate with `python scripts/make_track_a_scene.py`) |
| `third_party/` | the unmodified ROBOTIS TurtleBot3 Burger model (Apache 2.0) and the LDS-01 specification (MIT) |
| `tests/` | model, platform, structure and closed-loop tests |
| `results/demo/` | the demo run shown below |

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

Values that come from measurement are produced by scripts and stored in `config/`:

| Value | Script | Result |
|---|---|---|
| Marker range bias and noise model | `evaluation/vision_characterisation.py` | range bias corrected (residual mean -0.4 mm on an independent sweep), detection 94 to 99 percent at every range up to 3.6 m when the marker is fully in view |
| In-motion vision noise | `evaluation/vision_dynamic.py` | yaw noise 2.24 times the static model, position within the static model |
| Odometry calibration | `evaluation/odometry_calibration.py` | v scale 0.997, omega scale 0.992 (effective wheel separation 0.161 m) |
| EKF consistency tuning | `evaluation/tune_ekf.py` | process noise x5, yaw measurement std x2: mean NEES 2.99, 0.1 percent of updates gated out (12 settings x 10 runs, all docked) |

## Results

One run per start pad, seed 1, from `python scripts/run_demo.py --hidden` (`results/demo/summary.csv`). Docking
success requires the true final pose within 3 cm along the approach, 2 cm laterally and 5 degrees (these tolerances
are our assumption for a charging dock with spring contacts).

| Start pad | Marker visible at start | Result | Error along | Error lateral | Heading error | Time |
|---|---|---|---|---|---|---|
| north_west | yes | docked | -2.4 mm | 3.7 mm | -0.41 deg | 22.4 s |
| south_west | yes | docked | -1.0 mm | 3.4 mm | -0.53 deg | 25.8 s |
| south | yes | docked | 0.8 mm | -6.2 mm | 0.68 deg | 21.9 s |
| north | yes | docked | -2.7 mm | 5.2 mm | -0.58 deg | 16.8 s |
| centre | yes | docked | -0.9 mm | -2.8 mm | 0.37 deg | 26.2 s |
| far_west | no | docked | 0.7 mm | -6.8 mm | 0.80 deg | 49.2 s |
| far_north_west | no | docked | -3.5 mm | -11.6 mm | 1.18 deg | 50.8 s |

All seven runs docked. These are single runs; repeated trials under varied conditions are the next step.

## Known limitations

- **Start pads without a view of the marker are a risk.** Both hidden pads docked in this run, but from far_west the
  estimate drifted up to 140 mm while the marker was hidden. Over hidden stretches of 0.5 to 1 m, only 67 percent of
  the position errors were inside the filter's 3 sigma bound, so the filter is overconfident when it is blind for long.
  On the earlier Hakoniwa model the same start ended against the central obstacle. The LiDAR is scanned but not yet
  used for navigation or collision detection (Stage 2).
- **Results are single runs.** The Monte Carlo evaluation over varied conditions has not been run yet.
- **The filter is not equally consistent in every run.** Mean NEES ranges from 1.6 to 6.7 across the seven runs (3 is
  ideal); the north_west, south_west and far_west runs are above 4.
- **The position part of the measurement noise is conservative.** Median NIS is far below its expected value, because
  of a 5 mm floor on the position standard deviation.
- **Model patches and assumptions.** The three model patches above change the ROBOTIS model; the servo gain and the
  torque limit are our choices. The camera position, the odometry noise and the docking tolerances are assumptions.

## Contribution statement

| Member | Contribution |
|---|---|
| Ahmed El Zarw | State-space model, controllability and observability analysis, LQR tracker, frames and interfaces, platform wrapper, obstacle-aware path, evaluation harness, integration |
| TODO | EKF fusing wheel odometry with camera marker measurements (baseline in `src/estimation/`, to be reviewed and extended) |
| TODO | ArUco detection, camera calibration, docking trajectory (baseline in `src/vision/` and `src/docking/`, to be reviewed and extended) |

## Acknowledgements

- TurtleBot3 Burger MuJoCo model and meshes: ROBOTIS, `robotis_mujoco_menagerie`, Apache License 2.0
  (`third_party/robotis_mujoco_menagerie/`). The generated `models/tb3_burger.xml` is a modified copy.
- LDS-01 LiDAR specification: Hakoniwa Lab, `hakoniwa-mujoco-robots`, MIT licence (`third_party/hakoniwa-mujoco-robots/`).
  Its TurtleBot3 model was used until the switch to the ROBOTIS model and is no longer used.
- Linearisation, discretisation, LQR and EKF structure adapted from the weekly tutorials and from Ahmed El Zarw's
  41014 Assessment 3 work.
- The tracking-error model is the standard Kanayama form.

## Generative AI declaration

Tool: Claude (Anthropic), used through the Claude chat interface and Claude Code.

Purpose: Ahmed El Zarw specified the requirements, the design constraints and the order of work, and directed the
development. Claude Code generated most of the code and documentation in this repository under that direction. This
includes the platform wrapper, the models and the LQR tracker; the room scene, the camera rendering and the ArUco
measurement pipeline with its characterisation; the baseline EKF, docking and planning modules; the odometry
calibration and filter tuning; the mission logic; and the evaluation, demo and viewer scripts. Claude Code also ran the
diagnostics that found the platform issues listed above, including the three faults in the ROBOTIS model, and adapted
the project to that model.

Each team member is responsible for reviewing and understanding the material credited to them.
TODO before submission: each teammate adds their own tool use and purpose here, and the team records who reviewed which file.

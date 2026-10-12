"""Running, measuring and showing results. The only code allowed to read ground truth (to score the robot).

run_simulation.py         run_episode: one full run in MuJoCo, logs everything
true_position.py   the single access point to the true pose
scores.py        tracking errors, NEES, success-rate intervals
calibration/      offline calibration scripts that produce settings/ (vision, odometry, EKF tuning)
outputs/          plots, video, photos, the serial-monitor printout, reports
"""

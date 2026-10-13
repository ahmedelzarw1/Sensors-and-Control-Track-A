"""Running the robot and checking it against the truth (the only code allowed to read the true position).

    run_simulation.py               one full run in MuJoCo, logs everything
    true_position.py                the single place the true position is read
    scores.py                       errors, NEES, success-rate intervals
    plots.py                        room drawing, per-run CSV log, metrics and figures
    video.py, photos.py             demo video and camera photos
    serial_monitor.py               the serial-monitor printout of what the robot calculates
    calibrate_camera.py             marker range bias and noise model      -> settings/camera_noise.json
    camera_noise_while_driving.py   extra marker noise while driving       -> settings/camera_noise.json
    calibrate_wheels.py             odometry scales and process noise      -> settings/wheel_calibration.json
    tune_kalman_filter.py           filter consistency tuning              -> settings/kalman_filter_tuning.json
"""

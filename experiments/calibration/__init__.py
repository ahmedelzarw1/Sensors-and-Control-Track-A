"""Offline calibration. Each script measures against ground truth and writes a file in settings/ that the robot reads.

calibrate_camera.py   marker range bias and noise model R (settings/camera_noise.json)
camera_noise_while_driving.py            extra marker noise while driving (settings/camera_noise.json)
calibrate_wheels.py      odometry scale factors and process noise (settings/wheel_calibration.json)
tune_kalman_filter.py                  process and yaw noise scales for a consistent filter (settings/kalman_filter_tuning.json)
"""

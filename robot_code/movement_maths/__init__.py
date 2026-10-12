"""The maths of how the robot moves (the class topics: unicycle model, linearisation, controllability, observability).

unicycle_model.py
    f_unicycle, F_unicycle, G_unicycle   motion model and its Jacobians (used by the EKF)
    tracking_error, error_dynamics       error between the robot and the reference, in the robot frame
    linearise, zoh, error_model          state-space A, B and their discrete form (used by the LQR)
    ctrb, observability_matrix_ltv       controllability and observability checks
    H_marker_pose, H_marker_range_bearing   measurement Jacobians for the observability analysis
"""

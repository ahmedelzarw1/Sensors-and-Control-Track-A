"""Controller: turns "where I am" and "where I should be" into speed and turn commands.

lqr_controller.py    LQRTracker: feedforward + LQR feedback on the estimated tracking error (Bryson weights, Riccati gain)
smooth_commands.py   SlewRateLimiter: limits how fast the commands change
"""

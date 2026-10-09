"""Odometry calibration and process-noise measurement for the EKF.

Finding that motivates this (docs/estimation.md): on the simulated platform the wheel odometry over-reports rotation.
Driving arcs, the true heading change was 92 to 96 percent of the odometry value. The wheels scrub sideways and the
caster drags, so the effective wheel separation is larger than the geometric 0.16 m. Real robots show the same effect and
are calibrated for it (for example the UMBmark procedure). Here the calibration is done offline against simulator
ground truth, the simulated equivalent of a motion-capture calibration.

Procedure
1. Drive a fixed set of manoeuvres from rest (straight lines, arcs of both signs, turns in place), each through the
   slew-rate limiter the pipeline uses, with the default odometry noise.
2. Fit scale factors c_v and c_omega by least squares: true distance per step = c_v * v_odom * Ts and true heading
   change per step = c_omega * omega_odom * Ts.
3. With the calibrated odometry, compute the one-step residual between the true next pose and the exact unicycle
   prediction from the true current pose, in the robot frame (along, lateral, heading). Its covariance, minus the part
   already explained by the odometry noise, gives the extra process noise. Because the residuals are correlated in time
   (a slip lasts several steps), the per-step value is taken from the spread over 1 s windows divided by sqrt(10),
   which is larger than the raw per-step spread, so the filter is not overconfident.
4. The extra noise is split into a constant part and parts proportional to the distance and the angle moved.

Ground truth is used here only. Output: config/odometry_calibration.json (read by src/estimation/ekf.py) and
results/calibration/odometry_*.png|json.

Run: python -m evaluation.odometry_calibration
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.ground_truth import true_pose  # noqa: E402
from src.control.slew import SlewRateLimiter  # noqa: E402
from src.model.unicycle import f_unicycle  # noqa: E402
from src.tb3.interfaces import ControlCmd, wrap  # noqa: E402
from src.tb3.platform import DEFAULT_XML, OdomNoise, TB3Sim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "calibration"
CAL_JSON = ROOT / "config" / "odometry_calibration.json"
Ts = 0.1

# (v m/s, omega rad/s, seconds). Chosen to cover the speeds and turn rates the planner and docking produce.
MANOEUVRES = [(v, 0.0, 6.0) for v in (0.05, 0.10, 0.15, 0.20)] + \
             [(v, w, 5.0) for v in (0.06, 0.10, 0.15) for w in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8)] + \
             [(0.0, w, 3.0) for w in (-1.0, -0.5, -0.3, 0.3, 0.5, 1.0)]


def drive(seed: int = 0):
    """Odometry and true poses for every manoeuvre. Returns a list of (U (N,2), X (N+1,3))."""
    runs = []
    for i, (v, w, dur) in enumerate(MANOEUVRES):
        # The open-floor scene has the same robot, contacts and floor friction as the room, and space for every
        # manoeuvre without touching anything (the nearest obstacle is more than 2 m away from these paths).
        sim = TB3Sim(DEFAULT_XML, seed=seed + i, odom_noise=OdomNoise())
        sim.reset((-3.0, -2.0, 0.0))
        slew = SlewRateLimiter()
        U, X = [], [true_pose(sim)]
        n = int(dur / Ts)
        for k in range(n + 15):
            cmd = slew.apply(ControlCmd(v if k < n else 0.0, w if k < n else 0.0))
            obs = sim.step(cmd)
            U.append([obs.odom.v, obs.odom.omega])
            X.append(true_pose(sim))
        runs.append((np.array(U), np.array(X)))
    return runs


def step_truth(X):
    """True forward distance and heading change per step (robot frame)."""
    d = np.diff(X[:, :2], axis=0)
    c, s = np.cos(X[:-1, 2]), np.sin(X[:-1, 2])
    fwd = c * d[:, 0] + s * d[:, 1]
    dth = wrap(np.diff(X[:, 2]))
    return fwd, dth


def residuals(U, X, cv, cw):
    """One-step residual (along, lateral, heading) of the calibrated unicycle prediction from the true pose."""
    R = np.zeros((len(U), 3))
    for k in range(len(U)):
        pred = f_unicycle(X[k], (cv * U[k, 0], cw * U[k, 1]), Ts)
        e = X[k + 1] - pred
        c, s = np.cos(X[k, 2]), np.sin(X[k, 2])
        R[k] = [c * e[0] + s * e[1], -s * e[0] + c * e[1], wrap(e[2])]
    return R


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = drive()
    fwd = np.concatenate([step_truth(X)[0] for _, X in runs])
    dth = np.concatenate([step_truth(X)[1] for _, X in runs])
    Uall = np.concatenate([U for U, _ in runs])
    sv, sw = Uall[:, 0] * Ts, Uall[:, 1] * Ts
    cv = float(np.sum(fwd * sv) / np.sum(sv * sv))
    cw = float(np.sum(dth * sw) / np.sum(sw * sw))

    # Per-manoeuvre ratios, uncalibrated, for the table in docs/estimation.md.
    per = []
    for (v, w, dur), (U, X) in zip(MANOEUVRES, runs):
        f, t = step_truth(X)
        per.append({"v": v, "omega": w, "dist_ratio": float(f.sum() / (U[:, 0].sum() * Ts)) if v else None,
                    "turn_ratio": float(t.sum() / (U[:, 1].sum() * Ts)) if w else None})

    Rs = [residuals(U, X, cv, cw) for U, X in runs]
    Rraw = [residuals(U, X, 1.0, 1.0) for U, X in runs]
    # Windowed spread: sum residuals over 10-step windows, std / sqrt(10).
    def window_std(Rlist, n=10):
        sums = []
        for R in Rlist:
            for i in range(0, len(R) - n + 1, n):
                sums.append(R[i:i + n].sum(axis=0))
        return np.std(np.array(sums), axis=0) / np.sqrt(n)
    step_std = np.std(np.concatenate(Rs), axis=0)
    win_std = window_std(Rs)
    win_std_raw = window_std(Rraw)
    # Odometry-noise contribution per step (sigma_v Ts along, sigma_w Ts heading) is subtracted in quadrature.
    on = OdomNoise()
    expl = np.array([on.sigma_v * Ts, 0.0, on.sigma_omega * Ts])
    extra = np.sqrt(np.maximum(np.maximum(step_std, win_std) ** 2 - expl ** 2, 0.0))
    # Split into constant + proportional parts with a least-squares fit of |residual| on |motion| over all steps.
    Rall = np.concatenate(Rs)
    dist = np.abs(Uall[:, 0]) * Ts * cv
    turn = np.abs(Uall[:, 1]) * Ts * cw
    pos_res = np.hypot(Rall[:, 0], Rall[:, 1]) / np.sqrt(2)
    A = np.column_stack([np.ones_like(dist), dist])
    q_xy, k_dist = np.linalg.lstsq(A, np.abs(pos_res) * np.sqrt(np.pi / 2), rcond=None)[0]
    A = np.column_stack([np.ones_like(turn), turn])
    q_th, k_turn = np.linalg.lstsq(A, np.abs(Rall[:, 2]) * np.sqrt(np.pi / 2), rcond=None)[0]

    cal = {
        "source": "evaluation/odometry_calibration.py, %d manoeuvres, %d steps, default odometry noise" % (
            len(MANOEUVRES), len(Uall)),
        "scale_v": cv,
        "scale_omega": cw,
        "effective_wheel_separation_m": 0.16 / cw,
        "extra_noise_per_step": {
            "q_xy_m": float(max(q_xy, 0.0)),
            "k_dist": float(max(k_dist, 0.0)),
            "q_theta_rad": float(max(q_th, 0.0)),
            "k_turn": float(max(k_turn, 0.0)),
        },
        "residual_std_per_step_calibrated": step_std.tolist(),
        "residual_window_std_per_step_calibrated": win_std.tolist(),
        "residual_window_std_per_step_uncalibrated": win_std_raw.tolist(),
        "extra_std_after_odometry_noise": extra.tolist(),
    }
    CAL_JSON.parent.mkdir(exist_ok=True)
    CAL_JSON.write_text(json.dumps({k: cal[k] for k in ("source", "scale_v", "scale_omega", "extra_noise_per_step")},
                                   indent=2) + "\n")
    (OUT / "odometry_calibration.json").write_text(json.dumps({**cal, "per_manoeuvre": per}, indent=2) + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    tr = [p for p in per if p["turn_ratio"] is not None]
    ax[0].scatter([p["omega"] for p in tr], [p["turn_ratio"] for p in tr],
                  c=[p["v"] for p in tr], cmap="viridis")
    ax[0].axhline(cw, color="k", ls="--", label=f"fitted c_omega = {cw:.3f}")
    ax[0].set(xlabel="commanded omega [rad/s]", ylabel="true / odometry heading change",
              title="turn ratio per manoeuvre (colour: v)")
    ax[0].legend()
    dr = [p for p in per if p["dist_ratio"] is not None]
    ax[1].scatter([p["v"] for p in dr], [p["dist_ratio"] for p in dr])
    ax[1].axhline(cv, color="k", ls="--", label=f"fitted c_v = {cv:.3f}")
    ax[1].set(xlabel="commanded v [m/s]", ylabel="true / odometry distance", title="distance ratio per manoeuvre")
    ax[1].legend()
    ax[2].bar(np.arange(3) - 0.2, win_std_raw, 0.4, label="uncalibrated")
    ax[2].bar(np.arange(3) + 0.2, win_std, 0.4, label="calibrated")
    ax[2].set_xticks(range(3), ["along [m]", "lateral [m]", "heading [rad]"])
    ax[2].set(title="one-step residual std (1 s windows / sqrt 10)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "odometry_calibration.png", dpi=120)
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()

"""Characterise the ArUco marker measurement against ground truth: range-bias calibration and the covariance R.

Ground truth is used here only, to evaluate and calibrate the vision pipeline offline. The results are written to
settings/camera_noise.json, which robot_code/vision/marker_noise.py and robot_code/vision/marker_detection.py read. The robot never sees ground
truth at run time. This is the simulated equivalent of calibrating a camera against a surveyed target.

Method
- The robot is placed at random poses around the marker: planar distance d from the marker centre, viewing angle a
  (angle between the marker normal and the line from the marker to the robot), and heading offset b so that the
  marker appears at different places in the image. The room obstacles are moved out of the way for these sweeps
  only, so occlusion does not mix with detection range.
- For each pose: render, detect, solvePnP, convert to base_T_marker in B, compare with the true marker pose in B
  (true robot pose from experiments.true_position, marker body pose from MuJoCo).
- Pass 1 (calibration, seed 0, no correction): the camera-to-marker distance is biased long, growing with distance.
  The bias is fitted as e_r = k r^2 (r = camera-to-marker distance), which corresponds to the detected marker edges
  sitting a constant fraction of a pixel inside the true edges.
- Pass 2 (validation, seed 1, correction applied): an independent sweep checks that the bias is removed and gives
  the residual errors. The standard deviation of each component is fitted by maximum likelihood with
  sigma(d, a) = sqrt(c0^2 + (c1 d^2)^2) * (1 + c2 |a|). This is an empirical fit, not a physical law.
- Yaw errors above 15 degrees are counted as planar pose ambiguity flips, reported separately and left out of the
  fit. The EKF rejects them with its chi-square gate.

Run: python -m experiments.calibration.calibrate_camera            (both passes, about 8 minutes)
     python -m experiments.calibration.calibrate_camera --fit-only (refit from the saved sweeps)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from experiments.true_position import true_pose  # noqa: E402
from robot_code.room import room_map as km  # noqa: E402
from robot_code.simulated_robot.camera import SimCamera  # noqa: E402
from robot_code.data_types import ControlCmd, wrap  # noqa: E402
from robot_code.simulated_robot.simulator import ROOM_XML, OdomNoise, TB3Sim  # noqa: E402
from robot_code.vision.marker_detection import ArucoDetector, to_base_frame  # noqa: E402
from robot_code.vision.camera_maths import R_MJ_FROM_CV, horizontal_fov, intrinsics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "camera_calibration"
NOISE_JSON = ROOT / "settings" / "camera_noise.json"
OUTLIER_YAW = np.deg2rad(15.0)
KEYS = ["d", "view", "bearing", "in_frame", "detected", "ex", "ey", "eyaw", "range", "view_meas", "r_cam", "r_cam_true",
        "reproj", "reproj_alt"]


def marker_true(sim: TB3Sim) -> tuple[np.ndarray, np.ndarray]:
    """Marker planar pose (x, y, normal yaw) and 3D centre in W, read from the MuJoCo body (simulator truth)."""
    b = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "aruco_marker")
    R = sim.data.xmat[b].reshape(3, 3)
    normal = R[:, 2]  # plate local +z is the outward normal (see commands/build_mujoco_files.py)
    c = sim.data.xpos[b] + normal * (km.MARKER_PLATE_THICKNESS / 2)  # the printed face, not the plate centre
    return np.array([c[0], c[1], np.arctan2(normal[1], normal[0])]), c


def relative(robot: np.ndarray, marker: np.ndarray) -> np.ndarray:
    c, s = np.cos(robot[2]), np.sin(robot[2])
    dx, dy = marker[0] - robot[0], marker[1] - robot[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(marker[2] - robot[2]))])


def marker_in_frame(sim: TB3Sim, cam_id: int, margin_px: float = 2.0) -> bool:
    """True if all four outer corners of the black marker square project inside the image (simulator truth)."""
    b = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "aruco_marker")
    R = sim.data.xmat[b].reshape(3, 3)
    c = sim.data.xpos[b] + R[:, 2] * (km.MARKER_PLATE_THICKNESS / 2)
    h = km.MARKER_SIZE / 2
    K = intrinsics()
    Rc = sim.data.cam_xmat[cam_id].reshape(3, 3) @ R_MJ_FROM_CV  # OpenCV camera axes in W
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        pw = c + R[:, 0] * sx * h + R[:, 1] * sy * h
        pc = Rc.T @ (pw - sim.data.cam_xpos[cam_id])
        if pc[2] <= 0.0:
            return False
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        if not (margin_px <= u <= K[0, 2] * 2 - margin_px and margin_px <= v <= K[1, 2] * 2 - margin_px):
            return False
    return True


def clear_obstacles(sim: TB3Sim) -> None:
    for ob in km.OBSTACLES:
        g = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, ob.name)
        sim.model.geom_pos[g][2] = -5.0
    mujoco.mj_forward(sim.model, sim.data)


def sweep(n: int, seed: int, range_bias_k: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    sim = TB3Sim(ROOM_XML, odom_noise=OdomNoise(0.0, 0.0), lidar_noise=False)
    clear_obstacles(sim)
    cam = SimCamera(sim)
    det = ArucoDetector()
    extr = sim.sensor_extrinsics()["camera"]
    m_w, m_c = marker_true(sim)
    half_h = 0.5 * horizontal_fov()
    cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "color_camera")
    rows = []
    for _ in range(n):
        d = rng.uniform(0.18, 3.6)
        a = rng.uniform(-np.deg2rad(75), np.deg2rad(75))
        b = rng.uniform(-0.9, 0.9) * half_h
        p = m_w[:2] + d * np.array([np.cos(m_w[2] + a), np.sin(m_w[2] + a)])
        if not (km.ROOM_X_MIN + 0.15 < p[0] < km.ROOM_X_MAX - 0.15 and km.ROOM_Y_MIN + 0.15 < p[1] < km.ROOM_Y_MAX - 0.15):
            continue
        heading = np.arctan2(m_w[1] - p[1], m_w[0] - p[0]) - b  # marker appears at bearing about b
        sim.reset((p[0], p[1], heading))
        sim.step(ControlCmd(0.0, 0.0))  # let the contacts settle for one control period
        z_true = relative(true_pose(sim), m_w)
        dets = [q for q in det.detect(cam.render()) if q.marker_id == km.MARKER_ID]
        row = {"d": d, "view": a, "bearing": b, "in_frame": int(marker_in_frame(sim, cam_id)),
               "detected": int(bool(dets))}
        if dets:
            pm = to_base_frame(dets[0], extr, range_bias_k)
            e = pm.base_T_marker - z_true
            e[2] = wrap(e[2])
            r_cam = float(np.linalg.norm(dets[0].t_cam_marker))
            r_cam_true = float(np.linalg.norm(m_c - sim.data.cam_xpos[cam_id]))
            row.update(ex=e[0], ey=e[1], eyaw=e[2], range=pm.range, view_meas=pm.view_angle, r_cam=r_cam,
                       r_cam_true=r_cam_true, reproj=dets[0].reproj_err, reproj_alt=dets[0].reproj_err_alt)
        rows.append(row)
    cam.close()
    return rows


def save_rows(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in KEYS})


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return [{k: (float(v) if k not in ("detected", "in_frame") else int(v)) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]


def fit_range_bias(rows: list[dict]) -> float:
    """Least-squares k in r_meas - r_true = k r_true^2 over detections without an ambiguity flip."""
    rd = [r for r in rows if r["detected"] and abs(r["eyaw"]) <= OUTLIER_YAW]
    r = np.array([q["r_cam_true"] for q in rd])
    e = np.array([q["r_cam"] - q["r_cam_true"] for q in rd])
    return float(np.sum(e * r ** 2) / np.sum(r ** 4))


def fit_sigma(d, a, e) -> np.ndarray:
    """Maximum-likelihood fit of sigma(d, a) = sqrt(c0^2 + (c1 d^2)^2) * (1 + c2 |a|) to zero-mean residuals e."""
    def nll(c):
        s = np.sqrt(c[0] ** 2 + (c[1] * d ** 2) ** 2) * (1 + abs(c[2]) * np.abs(a))
        return float(np.sum(0.5 * (e / s) ** 2 + np.log(s)))
    c0 = np.array([np.std(e) * 0.5, np.std(e) * 0.2, 0.5])
    r = minimize(nll, c0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    return np.abs(r.x)


def distance_table(rows: list[dict]) -> list[dict]:
    table = []
    for lo, hi in [(0.15, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.6)]:
        tot = [r for r in rows if lo <= r["d"] < hi]
        dets = [r for r in tot if r["detected"]]
        inl = [r for r in dets if abs(r["eyaw"]) <= OUTLIER_YAW]
        inf = [r for r in tot if r["in_frame"]]
        row = {"d_range_m": f"{lo:.2f}-{hi:.2f}", "n_poses": len(tot),
               "detect_rate": round(len(dets) / len(tot), 3) if tot else None,
               "n_in_frame": len(inf),
               "detect_rate_in_frame": round(sum(r["detected"] for r in inf) / len(inf), 3) if inf else None,
               "n_detected": len(dets), "n_flips": len(dets) - len(inl)}
        if len(inl) >= 5:
            E = np.array([[r["ex"], r["ey"], r["eyaw"]] for r in inl])
            row.update(mean_x_mm=round(1e3 * E[:, 0].mean(), 2), std_x_mm=round(1e3 * E[:, 0].std(), 2),
                       mean_y_mm=round(1e3 * E[:, 1].mean(), 2), std_y_mm=round(1e3 * E[:, 1].std(), 2),
                       mean_yaw_deg=round(float(np.rad2deg(E[:, 2].mean())), 3),
                       std_yaw_deg=round(float(np.rad2deg(E[:, 2].std())), 3))
        table.append(row)
    return table


def detection_map(rows):
    rows = [r for r in rows if r["in_frame"]]
    D = np.array([r["d"] for r in rows]); A = np.array([r["view"] for r in rows])
    det = np.array([r["detected"] for r in rows], bool)
    d_edges = np.arange(0.0, 3.8, 0.2)
    a_edges = np.deg2rad(np.arange(-75, 76, 15))
    rate = np.full((len(d_edges) - 1, len(a_edges) - 1), np.nan)
    for i in range(len(d_edges) - 1):
        for j in range(len(a_edges) - 1):
            sel = (D >= d_edges[i]) & (D < d_edges[i + 1]) & (A >= a_edges[j]) & (A < a_edges[j + 1])
            if sel.sum() >= 3:
                rate[i, j] = det[sel].mean()
    return d_edges, a_edges, rate


def plot(cal_rows, val_rows, k, coeffs, path: Path) -> None:
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
    d_edges, _, rate = detection_map(val_rows)
    im = ax[0].imshow(rate.T, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis",
                      extent=[d_edges[0], d_edges[-1], -75, 75])
    ax[0].set(xlabel="distance to marker d [m]", ylabel="viewing angle [deg]", title="detection rate, marker fully in frame")
    fig.colorbar(im, ax=ax[0])
    c = [r for r in cal_rows if r["detected"] and abs(r["eyaw"]) <= OUTLIER_YAW]
    rt = np.array([r["r_cam_true"] for r in c]); er = np.array([r["r_cam"] - r["r_cam_true"] for r in c])
    ax[1].scatter(rt, 1e3 * er, s=3, alpha=0.4)
    rr = np.linspace(0.15, 3.6, 100)
    ax[1].plot(rr, 1e3 * k * rr ** 2, "k", label=f"fit e = k r^2, k = {k:.5f} 1/m")
    ax[1].set(xlabel="true camera-to-marker distance r [m]", ylabel="range error [mm]",
              title="pass 1: range bias before correction")
    ax[1].legend(fontsize=8)
    v = [r for r in val_rows if r["detected"]]
    d = np.array([r["d"] for r in v]); E = np.array([[r["ex"], r["ey"], r["eyaw"]] for r in v])
    out = np.abs(E[:, 2]) > OUTLIER_YAW
    dd = np.linspace(0.2, 3.5, 100)
    for kk, name, lab in [(0, "x", "x error [mm]"), (1, "y", "y error [mm]"), (2, "yaw", "yaw error [deg]")]:
        scale = np.rad2deg(1) if kk == 2 else 1e3
        a = ax[kk + 2]
        a.scatter(d[~out], E[~out, kk] * scale, s=3, alpha=0.4, label="inliers")
        lim = 3 * np.abs(E[~out, kk] * scale).max()
        if out.any():
            a.scatter(d[out], np.clip(E[out, kk] * scale, -lim, lim), s=6, c="r", label="ambiguity flips (clipped)")
        cc = coeffs[name]
        s0 = np.sqrt(cc[0] ** 2 + (cc[1] * dd ** 2) ** 2)
        for ang, ls in [(0.0, "-"), (np.deg2rad(60), "--")]:
            s = s0 * (1 + cc[2] * ang) * scale
            a.plot(dd, 3 * s, "k" + ls, lw=1, label=f"3 sigma fit, view {np.rad2deg(ang):.0f} deg")
            a.plot(dd, -3 * s, "k" + ls, lw=1)
        a.set(xlabel="planar distance d [m]", title="pass 2 (corrected): " + lab)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(n: int = 4000, fit_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal_csv, val_csv = OUT / "sweep_pass1_uncorrected.csv", OUT / "sweep_pass2_corrected.csv"
    if fit_only:
        cal_rows = load_rows(cal_csv)
        k = fit_range_bias(cal_rows)
        val_rows = load_rows(val_csv)
    else:
        cal_rows = sweep(n, seed=0, range_bias_k=0.0)
        save_rows(cal_rows, cal_csv)
        k = fit_range_bias(cal_rows)
        val_rows = sweep(n, seed=1, range_bias_k=k)
        save_rows(val_rows, val_csv)

    v = [r for r in val_rows if r["detected"] and abs(r["eyaw"]) <= OUTLIER_YAW]
    d = np.array([r["d"] for r in v]); a = np.array([r["view"] for r in v])
    E = np.array([[r["ex"], r["ey"], r["eyaw"]] for r in v])
    coeffs = {name: fit_sigma(d, a, E[:, i] - E[:, i].mean()).tolist() for i, name in enumerate(["x", "y", "yaw"])}

    # Largest distance up to which at least 90 percent of poses with the whole marker in the image and |view| < 45 deg
    # are detected, scanning 0.2 m bins outwards from 0.2 m.
    D = np.array([r["d"] for r in val_rows]); A = np.array([r["view"] for r in val_rows])
    det = np.array([r["detected"] for r in val_rows], bool)
    inf = np.array([r["in_frame"] for r in val_rows], bool)
    edges = np.arange(0.2, 3.8, 0.2)
    ok_until = None
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (D >= lo) & (D < hi) & (np.abs(A) < np.deg2rad(45)) & inf
        if sel.sum() >= 5 and det[sel].mean() >= 0.9:
            ok_until = float(hi)
        elif sel.sum() >= 5:
            break

    n_det = int(sum(r["detected"] for r in val_rows))
    summary = {
        "source": f"experiments/calibration/calibrate_camera.py, pass 1 seed 0 ({len(cal_rows)} poses), "
                  f"pass 2 seed 1 ({len(val_rows)} poses), obstacles removed",
        "range_bias_k_per_m": k,
        "range_bias_model": "r_true = r_meas - k r_meas^2 along the camera ray (fitted in pass 1)",
        "noise_model": "sigma_i(d, a) = sqrt(c0^2 + (c1*d^2)^2) * (1 + c2*|a|), d planar range m, a viewing angle rad",
        "coeffs": coeffs,
        "residual_mean_pass2": [float(x) for x in E.mean(axis=0)],
        "n_detected_pass2": n_det,
        "n_flips_pass2": n_det - len(v),
        "flip_rule_yaw_deg": float(np.rad2deg(OUTLIER_YAW)),
        "detected_90pct_in_frame_view_lt45_up_to_m": ok_until,
        "detect_rate_in_frame_pass2": float(det[inf].mean()),
        "detect_rate_in_frame_by_view": {f"{lo}-{lo + 15}": round(float(det[inf & (np.abs(np.rad2deg(A)) >= lo) &
                                                                       (np.abs(np.rad2deg(A)) < lo + 15)].mean()), 3)
                                         for lo in range(0, 75, 15)},
        "table_pass1_uncorrected": distance_table(cal_rows),
        "table_pass2_corrected": distance_table(val_rows),
    }
    NOISE_JSON.parent.mkdir(exist_ok=True)
    # Keep what other scripts wrote to the same file (the in-motion factor from camera_noise_while_driving.py).
    cfg = json.loads(NOISE_JSON.read_text()) if NOISE_JSON.exists() else {}
    cfg.update({k_: summary[k_] for k_ in ("source", "range_bias_k_per_m", "range_bias_model", "noise_model", "coeffs")})
    NOISE_JSON.write_text(json.dumps(cfg, indent=2) + "\n")
    (OUT / "vision_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(cal_rows, val_rows, k, coeffs, OUT / "vision_errors.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(int(args[0]) if args else 4000, fit_only="--fit-only" in sys.argv)

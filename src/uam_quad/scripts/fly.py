"""Fly the quad + arm live in the MuJoCo viewer (or headless for a quick check).

The geometric controller holds a position setpoint while the arm runs a chosen motion.
Inside the viewer: drag with the mouse to orbit, double-click a body to select it,
Ctrl + right-drag on a selected body to push it (apply a disturbance by hand), Space to pause.

    python scripts/fly.py                              # stock F450 + OpenMANIPULATOR-X, hover, arm swinging
    python scripts/fly.py --arm still --waypoints      # fly a 1 m square with the arm held
    python scripts/fly.py --preset typical --amp 1.0 --freq 1.0
    python scripts/fly.py --headless --seconds 10      # no window; prints a summary
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from uam_quad import GeometricController, UAMConfig, f450_mrs, f450_stock, hover_trim, load_model, with_omx_arm
from uam_quad.controller import quat_to_R

Q0 = np.array([-0.6, 1.0, 0.4])


def setpoint(t, waypoints):
    if not waypoints:
        return np.array([0.0, 0.0, 1.0])
    corners = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1.3], [0, 1, 1.3]], dtype=float)
    k = int(t // 4) % 4
    return corners[k]


def arm_target(t, mode, amp, freq):
    if mode == "still":
        return Q0.copy()
    s = amp * np.sin(2 * np.pi * freq * t)
    return Q0 + (np.array([s, -s, 0.0]) if mode == "swing" else np.array([s, 0.0, -s]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["f450_stock", "f450_mrs", "typical"], default="f450_stock")
    ap.add_argument("--arm-model", choices=["omx", "capsule"], default="omx", help="omx = ROBOTIS OpenMANIPULATOR-X")
    ap.add_argument("--arm", choices=["swing", "reach", "still"], default="swing")
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--waypoints", action="store_true", help="fly a square instead of hovering")
    ap.add_argument("--no-com-ff", action="store_true", help="turn off the CoM feedforward")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    cfg = {"f450_stock": f450_stock, "f450_mrs": f450_mrs, "typical": UAMConfig}[args.preset]()
    if args.arm_model == "omx":
        cfg = with_omx_arm(cfg)
    model, cfg = load_model(cfg)
    n = cfg.n_rotors
    trim = hover_trim(model, cfg, Q0)
    data = mujoco.MjData(model)
    data.qpos[:], data.ctrl[:] = trim.qpos, trim.ctrl
    data.act[:n] = trim.act
    mujoco.mj_forward(model, data)
    ctrl = GeometricController(model, cfg, com_ff=not args.no_com_ff)
    servo_offset = trim.ctrl[n:] - Q0
    max_tilt, max_err = 0.0, 0.0
    p_ref = setpoint(0.0, args.waypoints).astype(float)  # smoothed setpoint (1st-order, 0.8 s)

    def step():
        nonlocal max_tilt, max_err, p_ref
        t = data.time
        p_ref += (setpoint(t, args.waypoints) - p_ref) * model.opt.timestep / 0.8
        p_d = p_ref
        thrusts, _ = ctrl(data, p_d)
        data.ctrl[:n] = thrusts
        data.ctrl[n:] = arm_target(t, args.arm, args.amp, args.freq) + servo_offset
        mujoco.mj_step(model, data)
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(quat_to_R(data.qpos[3:7])[2, 2], -1, 1))))
        if not args.waypoints:
            max_err = max(max_err, np.linalg.norm(data.qpos[0:3] - p_d))

    if args.headless:
        while data.time < args.seconds:
            step()
    else:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.cam.lookat[:] = data.qpos[0:3]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, -120, -20
            while v.is_running() and data.time < args.seconds:
                t0 = time.time()
                for _ in range(int(round(0.02 / model.opt.timestep))):  # 20 ms of physics per frame
                    step()
                v.sync()
                time.sleep(max(0.0, 0.02 - (time.time() - t0)))
    print(f"{args.preset}: {data.time:.1f} s simulated, max tilt {max_tilt:.2f} deg"
          + ("" if args.waypoints else f", max hover error {max_err * 100:.1f} cm"))


if __name__ == "__main__":
    main()

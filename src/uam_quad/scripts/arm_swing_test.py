"""Closed-loop disturbance test: hover with the geometric controller, then swing the arm.

Phases: 0-3 s hold, 3-8 s sinusoidal arm motion (joints 1 and 2 in antiphase), 8-12 s hold.
Reports base position error, base tilt, end-effector deviation and rotor saturation, and saves
a plot plus a rendered frame to <repo>/runs/uam_quad/.

    python scripts/arm_swing_test.py --rotors 4 --amp 0.5 --freq 0.5
"""
import argparse
import os

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uam_quad import GeometricController, UAMConfig, arm_qpos_index, f450_mrs, hover_trim, load_model  # noqa: E402
from uam_quad.paths import RUNS  # noqa: E402
from uam_quad.controller import quat_to_R  # noqa: E402

OUT = RUNS
Q0 = np.array([-0.6, 1.0, 0.4])
PRESET = "typical"
P_D = np.array([0.0, 0.0, 1.0])


def arm_ref(t, amp, freq):
    if 3.0 <= t < 8.0:
        s = amp * np.sin(2 * np.pi * freq * (t - 3.0))
        return Q0 + np.array([s, -s, 0.0])
    return Q0.copy()


def run(n_rotors, amp, freq, T=12.0, render=True, com_ff=True):
    model, cfg = load_model(f450_mrs() if PRESET == 'f450_mrs' else UAMConfig(n_rotors=n_rotors))
    trim = hover_trim(model, cfg, Q0, pos=P_D)
    data = mujoco.MjData(model)
    data.qpos[:] = trim.qpos
    data.ctrl[:] = trim.ctrl
    data.act[:n_rotors] = trim.act
    mujoco.mj_forward(model, data)
    ctrl = GeometricController(model, cfg, com_ff=com_ff)
    servo_offset = trim.ctrl[n_rotors:] - Q0  # gravity sag the servos must pre-load
    ee_site = model.site("ee").id
    qa = arm_qpos_index(model)

    log = {k: [] for k in ("t", "ex", "tilt", "ee", "thr", "q")}
    frame = None
    steps = int(T / model.opt.timestep)
    for k in range(steps):
        t = k * model.opt.timestep
        thrusts, _ = ctrl(data, P_D)
        data.ctrl[:n_rotors] = thrusts
        data.ctrl[n_rotors:] = arm_ref(t, amp, freq) + servo_offset
        mujoco.mj_step(model, data)
        R = quat_to_R(data.qpos[3:7])
        log["t"].append(t)
        log["ex"].append(data.qpos[0:3] - P_D)
        log["tilt"].append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        log["ee"].append(data.site_xpos[ee_site].copy())
        log["thr"].append(thrusts.copy())
        log["q"].append(data.qpos[qa].copy())
        if render and frame is None and t >= 4.5:
            try:
                with mujoco.Renderer(model, 480, 720) as r:
                    r.update_scene(data, camera="side")
                    frame = r.render()
            except Exception as e:  # no GL context available
                print(f"(render skipped: {e})")
                render = False
    log = {k: np.array(v) for k, v in log.items()}
    return model, cfg, trim, log, frame


def summarize(name, log, cfg):
    t = log["t"]
    swing = (t >= 3.0) & (t < 8.0)
    settle = t >= 10.0
    exn = np.linalg.norm(log["ex"], axis=1)
    ee_dev = np.linalg.norm(log["ee"] - log["ee"][int(2.9 / (t[1] - t[0]))], axis=1)
    sat = np.mean((log["thr"] <= 1e-6) | (log["thr"] >= cfg.thrust_max - 1e-6)) * 100
    print(f"{name:>17}: base err max {exn[swing].max()*100:5.1f} cm | tilt max {log['tilt'][swing].max():5.2f} deg"
          f" | base err after settle {exn[settle].max()*100:4.2f} cm | rotor saturation {sat:4.1f}% of samples"
          f" | thrust range {log['thr'].min():.2f}-{log['thr'].max():.2f} N")
    return exn, ee_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotors", type=int, default=4)
    ap.add_argument("--preset", choices=["typical", "f450_mrs"], default="typical",
                    help="typical = earlier F450 guess; f450_mrs = CTU MRS F450 parameters + meshes")
    ap.add_argument("--amp", type=float, default=0.5, help="arm swing amplitude [rad]")
    ap.add_argument("--freq", type=float, default=0.5, help="arm swing frequency [Hz]")
    args = ap.parse_args()
    global PRESET
    PRESET = args.preset
    os.makedirs(OUT, exist_ok=True)

    model, cfg, trim, log, frame = run(args.rotors, args.amp, args.freq)
    _, _, _, log0, _ = run(args.rotors, 0.0, args.freq, render=False)
    _, _, _, log_noff, _ = run(args.rotors, args.amp, args.freq, render=False, com_ff=False)
    m = model.body_subtreemass[model.body("base").id]
    print(f"\n{args.rotors}-rotor + 3-DoF arm: total mass {m:.3f} kg, hover thrusts {np.round(trim.ctrl[:args.rotors], 3)} N, "
          f"trim |qacc| {trim.qacc_norm:.1e}")
    summarize("arm still", log0, cfg)
    summarize("swing, no CoM ff", log_noff, cfg)
    exn, ee_dev = summarize("swing, CoM ff", log, cfg)

    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    t = log["t"]
    ax[0].plot(t, log["q"]); ax[0].set_ylabel("arm joints [rad]"); ax[0].legend(["q1", "q2", "q3"], loc="upper right")
    ax[1].plot(t, log["ex"] * 100); ax[1].set_ylabel("base pos error [cm]"); ax[1].legend(["x", "y", "z"], loc="upper right")
    ax[2].plot(t, log["tilt"]); ax[2].set_ylabel("base tilt [deg]")
    ax[3].plot(t, log["thr"]); ax[3].set_ylabel("rotor thrust [N]"); ax[3].set_xlabel("time [s]")
    for a in ax:
        a.axvspan(3, 8, color="0.9", zorder=0)
        a.grid(alpha=0.3)
    fig.suptitle(f"{args.rotors}-rotor + 3-DoF arm, geometric controller, arm swing {args.amp} rad @ {args.freq} Hz (shaded)")
    fig.tight_layout()
    tag = f"{args.preset}_{args.rotors}rotor_amp{args.amp}_f{args.freq}"
    fig.savefig(f"{OUT}/arm_swing_{tag}.png", dpi=110)
    if frame is not None:
        plt.imsave(f"{OUT}/frame_{tag}.png", frame)
    print(f"saved {OUT}/arm_swing_{tag}.png" + (f" and frame_{tag}.png" if frame is not None else ""))


if __name__ == "__main__":
    main()

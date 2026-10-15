"""Closed-loop whole-body MPC on the F450 + 3-DoF arm in a reach task scene.

Scene: a post with a target pad in front of the drone, plus a second target to the side.
Timeline: hold (0-3 s) -> EE to target A (3-10 s) -> EE to target B (10-16 s) -> back home (16-22 s).
The MPC runs at 50 Hz on the true MuJoCo state; physics at 500 Hz with rotor lag and stiff joint
servos that track the MPC's arm reference model.

    python scripts/mpc_task.py                  # stock F450, writes plot + summary
    python scripts/mpc_task.py --video          # also renders <repo>/runs/uam_quad/mpc_task_<preset>.mp4
    python scripts/mpc_task.py --preset f450_mrs --push 3.0   # add a 3 N sideways push at t=12 s
"""
import argparse
import json
import os

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uam_quad import UAMConfig, f450_mrs, f450_stock, hover_trim, load_model, with_omx_arm  # noqa: E402
from uam_quad.paths import RUNS  # noqa: E402
from uam_quad.controller import quat_to_R  # noqa: E402
from uam_quad.wbmpc import ArmReference, MPCParams, WholeBodyMPC, state_from_mujoco  # noqa: E402

OUT = RUNS
TH_NOM = np.array([-0.6, 1.0, 0.4])
START = np.array([0.0, 0.0, 1.0])
TARGET_A = np.array([1.2, 0.0, 1.35])
TARGET_B = np.array([1.2, 0.45, 1.15])

SCENE = f"""
    <body name="post" pos="{TARGET_A[0] + 0.03} {TARGET_A[1]} 0">
      <geom name="post" type="cylinder" fromto="0 0 0 0 0 {TARGET_A[2] - 0.05}" size="0.02" rgba="0.45 0.35 0.25 1"/>
      <geom name="pad" type="box" pos="0 0 {TARGET_A[2] - 0.045}" size="0.05 0.05 0.005" rgba="0.3 0.7 0.35 1"/>
    </body>
    <site name="target_A" pos="{TARGET_A[0]} {TARGET_A[1]} {TARGET_A[2]}" size="0.022" rgba="0.2 0.9 0.3 0.5"/>
    <site name="target_B" pos="{TARGET_B[0]} {TARGET_B[1]} {TARGET_B[2]}" size="0.022" rgba="0.95 0.75 0.1 0.5"/>
    <body name="target_mark" mocap="true" pos="0 0 -1"><geom type="sphere" size="0.012" rgba="1 0.2 0.2 0.9" contype="0" conaffinity="0"/></body>"""


def phase(t, ee_home):
    if t < 3.0:
        return "hold", ee_home
    if t < 10.0:
        return "reach A", TARGET_A
    if t < 16.0:
        return "reach B", TARGET_B
    return "return", ee_home


SEGMENTS = [(3.0, "A"), (10.0, "B"), (16.0, "home")]
MAX_SPEED = 0.45  # m/s average along each move; move time = max(MIN_MOVE, distance / MAX_SPEED)
MIN_MOVE = 2.5


def min_jerk(p0, p1, T, tau):
    s = np.clip(tau / T, 0.0, 1.0)
    pos = p0 + (p1 - p0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    vel = (p1 - p0) * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T if 0.0 < tau < T else np.zeros(3)
    return pos, vel


def ee_reference(t, ee_home):
    """Smooth EE reference (position, velocity): minimum-jerk moves between the task targets."""
    pts = {"A": TARGET_A, "B": TARGET_B, "home": ee_home}
    prev, pos, vel = ee_home, ee_home, np.zeros(3)
    for t0, key in SEGMENTS:
        if t < t0:
            break
        nxt = pts[key]
        T = max(MIN_MOVE, np.linalg.norm(nxt - prev) / MAX_SPEED)
        pos, vel = min_jerk(prev, nxt, T, t - t0)
        prev = nxt
    return pos, vel


SWEEP_F = 1.0


SWEEP_CENTER = np.array([-0.03, 0.0, 0.0])  # figure centred inside the OMX workspace (all points reachable, base fixed)


def sweep_ref(t, ee_home, f=None, ax=0.05, az=0.05):
    """EE figure-eight in the arm plane (x-z) around ee_home + SWEEP_CENTER, ramped in/out over 1 s, active 2-12 s."""
    if t < 2.0 or t >= 12.0:
        return ee_home.copy()
    f = SWEEP_F if f is None else f
    ramp = min(1.0, (t - 2.0), (12.0 - t))
    w = 2 * np.pi * f * (t - 2.0)
    return ee_home + ramp * (SWEEP_CENTER + np.array([ax * np.sin(w), 0.0, az * np.sin(2 * w)]))


def write_video(frames, path):
    if not frames:
        return
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    out = []
    for img, t, name, e, tilt, sms, f in frames:
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, 960, 70], fill=(19, 23, 31, 215))
        d.text((14, 8), f"Whole-body MPC · F450 + OpenMANIPULATOR-X (3-DoF) · {name}", font=fb, fill=(238, 236, 230))
        d.text((14, 40), f"t {t:4.1f} s   EE error {e * 100:5.1f} cm   tilt {tilt:4.1f}°   solve {sms:4.1f} ms   thrust {f.min():.1f}–{f.max():.1f} N",
               font=fnt, fill=(242, 154, 74))
        out.append(np.asarray(im))
    imageio.mimwrite(path, out, fps=30, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=8)
    print("video", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["f450_stock", "f450_mrs", "typical"], default="f450_stock")
    ap.add_argument("--arm", choices=["omx", "capsule"], default="omx", help="omx = ROBOTIS OpenMANIPULATOR-X (CAD masses)")
    ap.add_argument("--T", type=float, default=22.0)
    ap.add_argument("--push", type=float, default=0.0, help="sideways push [N] for 0.5 s at t=12 s")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--task", choices=["reach", "sweep"], default="reach",
                    help="sweep = base holds position while the EE traces a fast 1 Hz figure in the arm plane")
    ap.add_argument("--no-coupling", action="store_true", help="ablation: MPC model without arm-motion coupling terms")
    ap.add_argument("--arm-wn", type=float, default=12.0, help="arm reference-model bandwidth [rad/s]")
    ap.add_argument("--w-base", type=float, default=2.0, help="weight on the base-position anchor")
    ap.add_argument("--sweep-f", type=float, default=1.0, help="sweep frequency [Hz]")
    ap.add_argument("--servo-kp", type=float, default=None, help="override joint servo stiffness [N*m/rad]")
    ap.add_argument("--servo-kv", type=float, default=None)
    ap.add_argument("--fast", action="store_true", help="faster moves (1.0 m/s, 1.2 s min) to stress the arm coupling")
    args = ap.parse_args()

    cfg = {"f450_stock": f450_stock, "f450_mrs": f450_mrs, "typical": UAMConfig}[args.preset]()
    if args.arm == "omx":
        cfg = with_omx_arm(cfg)
    if args.servo_kp is not None:
        cfg.servo_kp = args.servo_kp
    if args.servo_kv is not None:
        cfg.servo_kv = args.servo_kv
    model, cfg = load_model(cfg, extra_world=SCENE)
    n = cfg.n_rotors
    dt = model.opt.timestep
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    qv = [model.jnt_dofadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    ee_site, base_id = model.site("ee").id, model.body("base").id
    mark = model.body("target_mark").mocapid[0]

    trim = hover_trim(model, cfg, TH_NOM, pos=START)
    data = mujoco.MjData(model)
    data.qpos[:], data.ctrl[:] = trim.qpos, trim.ctrl
    data.act[:n] = trim.act
    mujoco.mj_forward(model, data)
    ee_home = data.site_xpos[ee_site].copy()
    ee_offset = ee_home - START  # EE relative to base at the nominal arm pose (yaw 0)

    prm = MPCParams(coupling=not args.no_coupling, arm_wn=args.arm_wn, w_base=args.w_base)
    global SWEEP_F
    SWEEP_F = args.sweep_f
    global MAX_SPEED, MIN_MOVE
    if args.fast:
        MAX_SPEED, MIN_MOVE = 1.0, 1.2
    mpc = WholeBodyMPC(model, cfg, prm, th_nominal=TH_NOM)
    u_hover = np.r_[trim.ctrl[:n], TH_NOM]
    x0 = state_from_mujoco(model, data, qa, qv)
    mpc.warm_start(x0, u_hover)
    arm_ref = ArmReference(TH_NOM, prm.arm_wn)
    kp = cfg.servo_kp

    every = int(round(0.02 / dt))  # MPC at 50 Hz
    log = {k: [] for k in ("t", "ee", "ee_ref", "base", "tilt", "f", "th", "solve_ms", "status")}
    frames, renderer, cam = [], None, None
    if args.video:
        renderer = mujoco.Renderer(model, 540, 960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 1.55, -128, -14
    u = u_hover.copy()
    current = None
    for k in range(int(args.T / dt)):
        t = k * dt
        name, ee_ref = phase(t, ee_home)
        if name != current:
            current = name
            data.mocap_pos[mark] = ee_ref  # final target of this phase (red marker)
        if args.task == "sweep":
            name, ee_ref = ("sweep" if 2.0 <= t < 12.0 else "hold"), sweep_ref(t, ee_home)
        if k % every == 0:
            if args.task == "sweep":
                ee_tr = np.array([sweep_ref(t + j * prm.dt, ee_home) for j in range(prm.N + 1)])
                base_tr, vel_tr = np.tile(START, (prm.N + 1, 1)), np.zeros((prm.N + 1, 3))
            else:
                refs = [ee_reference(t + j * prm.dt, ee_home) for j in range(prm.N + 1)]
                ee_tr = np.array([r[0] for r in refs])
                vel_tr = np.array([r[1] for r in refs])
                base_tr = ee_tr - ee_offset
            mpc.set_trajectory(ee_tr, base_tr, vel_tr, u_hover, th_ref=TH_NOM)
            u, status = mpc(state_from_mujoco(model, data, qa, qv))
            log["solve_ms"].append(mpc.solve_times[-1] * 1e3)
            log["status"].append(status)
        # actuators: rotor thrust commands (plant adds motor lag); servos track the arm reference
        data.ctrl[:n] = np.clip(u[:n], 0.0, cfg.thrust_max)
        th_r, _ = arm_ref.step(u[n:], dt)
        data.ctrl[n:] = th_r + data.qfrc_bias[qv] / kp  # servo with gravity/Coriolis feedforward
        data.xfrc_applied[base_id, :3] = [0.0, args.push, 0.0] if 12.0 <= t < 12.5 else 0.0
        mujoco.mj_step(model, data)

        if k % 10 == 0:
            R = quat_to_R(data.qpos[3:7])
            log["t"].append(t)
            log["ee"].append(data.site_xpos[ee_site].copy())
            log["ee_ref"].append(ee_ref.copy())
            log.setdefault("ee_traj", []).append(ee_reference(t, ee_home)[0])
            log["base"].append(data.qpos[0:3].copy())
            log["tilt"].append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
            log["f"].append(data.act[:n].copy())
            log["th"].append(data.qpos[qa].copy())
            log.setdefault("thd", []).append(data.qvel[qv].copy())
            log.setdefault("th_ref", []).append(th_r.copy())
        if args.video and k % int(round(1 / (30 * dt))) == 0:
            cam.lookat[:] = 0.5 * (data.qpos[0:3] + data.site_xpos[ee_site])
            renderer.update_scene(data, camera=cam)
            frames.append((renderer.render(), t, name, np.linalg.norm(data.site_xpos[ee_site] - ee_ref), log["tilt"][-1],
                           mpc.solve_times[-1] * 1e3, data.act[:n].copy()))

    L = {k: np.array(v) for k, v in log.items()}
    err = np.linalg.norm(L["ee"] - L["ee_ref"], axis=1)
    track = np.linalg.norm(L["ee"] - L["ee_traj"], axis=1)
    if args.task == "sweep":
        sel = (L["t"] >= 3.0) & (L["t"] < 11.0)
        base_dev = np.linalg.norm(L["base"] - START, axis=1)
        over = np.where(L["tilt"] > 10.0)[0]
        t_over = float(L["t"][over[0]]) if len(over) else None
        pre = (L["t"] < (t_over if t_over else 99)) & (L["t"] >= 2.0)
        summary_extra = {"t_tilt_over_10deg": t_over,
                         "max_joint_speed_before_rad_s": round(float(np.abs(L["thd"][pre]).max()), 2) if pre.any() else None}
        summary = {"task": "sweep", "coupling": prm.coupling, **summary_extra, "arm_wn": prm.arm_wn, "w_base": prm.w_base, "sweep_f": SWEEP_F,
                   "preset": args.preset, "arm": args.arm,
                   "ee_track_rms_mm": round(float(np.sqrt(np.mean(err[sel] ** 2)) * 1e3), 2),
                   "ee_track_max_mm": round(float(err[sel].max() * 1e3), 2),
                   "base_dev_rms_mm": round(float(np.sqrt(np.mean(base_dev[sel] ** 2)) * 1e3), 2),
                   "max_tilt_deg": round(float(L["tilt"][sel].max()), 2),
                   "arm_joint_range_rad": [round(float(v), 2) for v in np.ptp(L["th"][sel], axis=0)],
                   "servo_lag_rms_rad": [round(float(v), 4) for v in np.sqrt(np.mean((L["th"][sel] - L["th_ref"][sel]) ** 2, axis=0))],
                   "solve_ms_median": round(float(np.median(L["solve_ms"])), 2), "solver_status": dict(mpc.status_counts)}
        print(json.dumps(summary, indent=1))
        tag = f"mpc_sweep_{args.preset}_{args.arm}_wn{prm.arm_wn:g}_wb{prm.w_base:g}_f{SWEEP_F:g}" + ("_nocpl" if args.no_coupling else "")
        json.dump(summary, open(f"{OUT}/{tag}.json", "w"), indent=1)
        write_video(frames, f"{OUT}/{tag}.mp4")
        return
    summary = {"preset": args.preset, "arm": args.arm, "coupling": prm.coupling, "fast": args.fast, "mass_kg": round(float(mpc.wb.mass), 3)}
    for name, (a, b) in {"hold": (2.0, 3.0), "reach A": (8.0, 10.0), "reach B": (14.0, 16.0), "return": (20.0, 22.0)}.items():
        sel = (L["t"] >= a) & (L["t"] < b)
        summary[f"ee_err_{name}_mm"] = round(float(err[sel].mean() * 1e3), 1)
    for name, t0, t1 in (("A", 3.0, 10.0), ("B", 10.0, 16.0)):
        sel = (L["t"] >= t0) & (L["t"] < t1)
        tt, ee_ = L["t"][sel], err[sel]
        inside = np.where(ee_ > 0.02)[0]
        summary[f"settle_2cm_{name}_s"] = round(float(tt[inside[-1]] - t0), 2) if len(inside) else 0.0
    moving = ((L["t"] >= 3) & (L["t"] < 9)) | ((L["t"] >= 10) & (L["t"] < 13)) | ((L["t"] >= 16) & (L["t"] < 22))
    summary["ee_track_err_while_moving_mm_rms"] = round(float(np.sqrt(np.mean(track[moving] ** 2)) * 1e3), 1)
    summary["ee_track_err_while_moving_mm_max"] = round(float(track[moving].max() * 1e3), 1)
    summary.update(max_tilt_deg=round(float(L["tilt"].max()), 1), thrust_min_N=round(float(L["f"].min()), 2),
                   thrust_max_N=round(float(L["f"].max()), 2), thrust_limit_N=round(cfg.thrust_max, 2),
                   solve_ms_median=round(float(np.median(L["solve_ms"])), 2), solve_ms_max=round(float(L["solve_ms"].max()), 2),
                   solver_status=dict(mpc.status_counts), push_N=args.push)
    print(json.dumps(summary, indent=1))
    os.makedirs(OUT, exist_ok=True)
    tag = (f"mpc_task_{args.preset}_{args.arm}" + (f"_push{args.push:g}" if args.push else "")
           + ("_nocpl" if args.no_coupling else "") + ("_fast" if args.fast else ""))
    json.dump(summary, open(f"{OUT}/{tag}.json", "w"), indent=1)

    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for i, c in enumerate("xyz"):
        ax[0].plot(L["t"], L["ee"][:, i], label=f"EE {c}")
        ax[0].plot(L["t"], L["ee_traj"][:, i], "--", color=f"C{i}", alpha=0.6)
    ax[0].set_ylabel("EE position [m]"); ax[0].legend(ncol=3, loc="upper left")
    ax[1].semilogy(L["t"], np.maximum(err, 1e-4), label="to final target")
    ax[1].semilogy(L["t"], np.maximum(track, 1e-4), label="to smooth reference")
    ax[1].axhline(0.02, color="0.5", ls=":"); ax[1].set_ylabel("EE error [m]"); ax[1].legend(loc="upper right")
    ax[2].plot(L["t"], L["tilt"]); ax[2].set_ylabel("base tilt [deg]")
    ax[3].plot(L["t"], L["f"]); ax[3].axhline(cfg.thrust_max, color="r", ls=":"); ax[3].set_ylabel("rotor thrust [N]"); ax[3].set_xlabel("time [s]")
    for a in ax:
        for x_ in (3, 10, 16):
            a.axvline(x_, color="0.8", lw=1)
        a.grid(alpha=0.3)
    fig.suptitle(f"Whole-body MPC, {args.preset} + {'OpenMANIPULATOR-X' if args.arm == 'omx' else 'capsule'} 3-DoF arm: reach A (3 s), B (10 s), home (16 s)" + (f", {args.push:g} N push at 12 s" if args.push else ""))
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}.png", dpi=110)

    write_video(frames, f"{OUT}/{tag}.mp4")


if __name__ == "__main__":
    main()

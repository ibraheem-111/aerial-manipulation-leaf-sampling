"""Whole-body MPC vs cascaded PID (+ arm IK) on an arm-driven task, ideal and realistic conditions.

Task (base anchored at a standoff after the approach, so the ARM has to move the gripper):
  0-2 s   hover at the start
  2-7 s   approach: fly to a standoff in front of the post (gripper 6 cm short of the pad)
  7-9 s   arm reaches the pad
  9-21 s  arm-only motion: gripper visits points around the pad (base should stay put)
  21-24 s hold on the pad

Controllers
  mpc : whole-body MPC (uam_quad.wbmpc), nominal model, plus integral action on the EE error
  pid : cascaded PID as on real autopilots: position PID -> geometric attitude PD (uam_quad.controller)
        for the base, and inverse kinematics from the *measured* base pose for the arm (the decoupled
        "IK + cascaded PID" baseline of Flying Hand).

Realistic mode (identical for both controllers)
  sensors   : position 2 mm, velocity 2 cm/s, attitude 0.3 deg, gyro 0.02 rad/s, joints 0.002 rad (1-sigma)
  latency   : the controller sees the state 10 ms late
  mismatch  : plant airframe +10 % mass, rotors give 95 % of commanded thrust, motor lag 30 ms (model: 20 ms)
  wind      : Ornstein-Uhlenbeck force on the airframe, mean (0.3, 0.15, 0) N, sigma 0.5 N, tau 1.5 s

    python scripts/compare_controllers.py                 # 4 runs, table + plots
    python scripts/compare_controllers.py --video         # + side-by-side realistic video
"""
import argparse
import json
import os

import matplotlib
import mujoco
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uam_quad import Gains, GeometricController, f450_stock, hover_trim, load_model, with_omx_arm  # noqa: E402
from uam_quad.paths import RUNS, BUILD  # noqa: E402
from uam_quad.estimator import DisturbanceKF  # noqa: E402
from uam_quad.lqr import LQIController  # noqa: E402
from uam_quad.controller import quat_to_R  # noqa: E402
from uam_quad.eom import ArmBodyParams, arm_frames  # noqa: E402
from uam_quad.wbmpc import ArmReference, MPCParams, WholeBodyMPC, state_from_mujoco  # noqa: E402

OUT = os.path.join(RUNS, "compare")
TH_NOM = np.array([-0.6, 1.0, 0.4])
START = np.array([0.0, 0.0, 1.0])
PAD = np.array([1.0, 0.0, 1.30])              # gripper target on the pad
STANDOFF_GAP = np.array([-0.06, 0.0, 0.0])    # where the gripper is (nominal arm) when the base parks
ARM_PTS = [np.array(v) for v in ([0, 0, 0.06], [-0.05, 0, 0.06], [-0.05, 0, -0.04], [0.0, 0, -0.04], [0.0, 0, 0.0],
                                 [-0.06, 0, 0.03], [0.0, 0, 0.0])]   # offsets from the pad, in the arm plane
SCENE = f"""
    <body name="post" pos="{PAD[0] + 0.035} {PAD[1]} 0">
      <geom name="post" type="cylinder" fromto="0 0 0 0 0 {PAD[2] - 0.05}" size="0.02" rgba="0.45 0.35 0.25 1"/>
      <geom name="pad" type="box" pos="0 0 {PAD[2] - 0.045}" size="0.05 0.05 0.005" rgba="0.3 0.7 0.35 1" contype="0" conaffinity="0"/>
    </body>
    <body name="target_mark" mocap="true" pos="0 0 -1"><geom type="sphere" size="0.01" rgba="1 0.2 0.2 0.9" contype="0" conaffinity="0"/></body>"""


def min_jerk(p0, p1, T, tau):
    s = np.clip(tau / T, 0.0, 1.0)
    b = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    db = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T if 0 < tau < T else 0.0
    return p0 + (p1 - p0) * b, (p1 - p0) * db


class TaskRef:
    """Reference for the gripper (EE) and the base, as functions of time."""

    def __init__(self, ee_offset):
        self.off = ee_offset                       # EE - base at the nominal arm pose
        self.base_park = PAD + STANDOFF_GAP - ee_offset
        self.ee_home = START + ee_offset
        segs, t = [], 9.0
        prev = PAD.copy()
        for d in ARM_PTS:
            nxt = PAD + d
            segs.append((t, prev, nxt, 1.6))
            prev, t = nxt, t + 1.7
        self.arm_segs = segs                       # ends at ~20.9 s

    def __call__(self, t):
        """(ee_pos, base_pos, base_vel, phase name)."""
        if t < 2.0:
            return self.ee_home, START, np.zeros(3), "hover"
        if t < 7.0:
            b, bv = min_jerk(START, self.base_park, 5.0, t - 2.0)
            return b + self.off, b, bv, "approach"
        if t < 9.0:
            e, _ = min_jerk(self.base_park + self.off, PAD, 1.6, t - 7.0)
            return e, self.base_park, np.zeros(3), "reach pad"
        for t0, p0, p1, T in reversed(self.arm_segs):
            if t >= t0:
                e, _ = min_jerk(p0, p1, T, t - t0)
                return e, self.base_park, np.zeros(3), ("arm moves" if t < 21.0 else "hold on pad")
        return PAD, self.base_park, np.zeros(3), "reach pad"


class Sensors:
    def __init__(self, realistic, rng, delay_steps):
        self.real, self.rng = realistic, rng
        self.buf, self.delay = [], delay_steps

    def __call__(self, x):
        """x = MPC-format state (19). Returns the noisy, delayed estimate."""
        x = x.copy()
        if self.real:
            r = self.rng
            x[0:3] += r.normal(0, 0.002, 3)
            dq = r.normal(0, np.radians(0.3), 3)
            q = x[3:7]
            dqq = np.r_[1.0, 0.5 * dq]
            q = np.array([q[0] * dqq[0] - q[1:] @ dqq[1:], *(q[0] * dqq[1:] + dqq[0] * q[1:] + np.cross(q[1:], dqq[1:]))])
            x[3:7] = q / np.linalg.norm(q)
            x[7:10] += r.normal(0, 0.02, 3)
            x[10:13] += r.normal(0, 0.02, 3)
            x[13:16] += r.normal(0, 0.002, 3)
            x[16:19] += r.normal(0, 0.02, 3)
        self.buf.append(x)
        if len(self.buf) > self.delay + 1:
            self.buf.pop(0)
        return self.buf[0] if self.real else x


def ik(P, target_B, th0, lo, hi):
    f = lambda th: arm_frames(P, th)[0][3][[0, 2]] - target_B[[0, 2]]  # noqa: E731
    return least_squares(f, th0, bounds=(lo, hi), xtol=1e-6).x


PID_TUNED = dict(pos_wn=3.0, pos_ki=0.8, i_limit=1.0)   # chosen by scripts/sweep_wind.py --tune (only robust setting, used with pid_tuned_noouter)
WIND_DIR = np.array([0.3, 0.15, 0.0]) / np.linalg.norm([0.3, 0.15, 0.0])


def run(ctrl_name, realistic, video=False, seed=0, wind_mag=0.335, pid_gains=None):
    rng = np.random.default_rng(seed)
    nominal = with_omx_arm(f450_stock())
    plant_cfg = with_omx_arm(f450_stock(base_mass=nominal.base_mass * (1.10 if realistic else 1.0),
                                        motor_tau=0.03 if realistic else nominal.motor_tau))
    model, cfg = load_model(plant_cfg, extra_world=SCENE)        # the plant
    cmodel, ccfg = load_model(nominal, extra_world=SCENE)         # what the controller believes
    if realistic:
        model.actuator_gainprm[:4, 0] *= 0.95                     # props give 95 % of commanded thrust
    n, dt = cfg.n_rotors, model.opt.timestep
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    qv = [model.jnt_dofadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    ee_site, base_id, mark = model.site("ee").id, model.body("base").id, model.body("target_mark").mocapid[0]

    trim_plant = hover_trim(model, cfg, TH_NOM, pos=START)
    trim_nom = hover_trim(cmodel, ccfg, TH_NOM, pos=START)
    data = mujoco.MjData(model)
    data.qpos[:], data.ctrl[:] = trim_plant.qpos, trim_plant.ctrl
    data.act[:n] = trim_plant.act
    mujoco.mj_forward(model, data)
    ref = TaskRef(data.site_xpos[ee_site] - START)
    est = mujoco.MjData(cmodel)                                   # controller-side copy for the PID baseline
    sensors = Sensors(realistic, rng, delay_steps=5)
    P = ArmBodyParams(cmodel, ccfg)
    lo = np.array([r[0] for r in ccfg.joint_ranges]); hi = np.array([r[1] for r in ccfg.joint_ranges])

    prm = MPCParams(arm_wn=25.0, w_base=200.0, codegen_dir=os.path.join(BUILD, f"uam_wbmpc_{os.getpid()}"))
    u_hover = np.r_[trim_nom.ctrl[:n], TH_NOM]
    if ctrl_name in ("mpc", "mpc_kf", "mpc_noint"):
        mpc = WholeBodyMPC(cmodel, ccfg, prm, th_nominal=TH_NOM)
        kf = DisturbanceKF(mpc.wb, dt=0.02, motor_tau=ccfg.motor_tau) if ctrl_name == "mpc_kf" else None
        mpc.warm_start(state_from_mujoco(model, data, qa, qv), u_hover)
        ee_int = np.zeros(3)
    else:
        if ctrl_name == "pid":
            geo = GeometricController(cmodel, ccfg)                              # default gains (pos_ki 0.6)
        elif ctrl_name == "pid_tuned":           # stronger position loop + integrator (plus the shared outer EE integrator)
            geo = GeometricController(cmodel, ccfg, Gains(**(pid_gains or PID_TUNED)))
        elif ctrl_name == "pid_tuned_noouter":   # same, without the outer EE integrator
            geo = GeometricController(cmodel, ccfg, Gains(**(pid_gains or PID_TUNED)))
        elif ctrl_name == "lqr":                 # one gain, linearized at hover with the nominal arm pose
            lqr = LQIController(cmodel, ccfg, TH_NOM, pos=START)
        elif ctrl_name == "lqr_armblind":        # same, but no feedback on arm states (arm treated as a disturbance)
            lqr = LQIController(cmodel, ccfg, TH_NOM, pos=START, arm_feedback=False)
        elif ctrl_name == "lqr_sched":           # gain-scheduled on the arm pose (relinearize every 0.15 rad), arm states not fed back
            lqr = LQIController(cmodel, ccfg, TH_NOM, pos=START, schedule_step=0.15, arm_feedback=False)
        elif ctrl_name == "lqr_sched_armfb":     # scheduled AND linear feedback on arm states
            lqr = LQIController(cmodel, ccfg, TH_NOM, pos=START, schedule_step=0.15)
        else:
            raise ValueError(ctrl_name)
        th_ik = TH_NOM.copy()
        ee_int = np.zeros(3)
    arm_ref = ArmReference(TH_NOM, prm.arm_wn)
    wind_mean, wind = wind_mag * WIND_DIR, np.zeros(3)
    wind_sigma = 1.5 * wind_mag                                                  # 0.5 N at the default 0.335 N

    every = int(round(0.02 / dt))
    u = u_hover.copy()
    log = {k: [] for k in ("t", "ee", "ee_ref", "base", "base_ref", "tilt", "f", "th", "phase")}
    frames = []
    renderer = cam = None
    if video:
        renderer = mujoco.Renderer(model, 540, 720)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 0.95, -112, -10
    for k in range(int(24.0 / dt)):
        t = k * dt
        ee_r, base_r, base_v, ph = ref(t)
        x_meas = sensors(state_from_mujoco(model, data, qa, qv))
        if not ctrl_name.startswith("mpc"):   # autopilot-style: base loop every physics step (500 Hz), arm IK at 50 Hz
            est.qpos[:] = 0; est.qpos[0:7] = x_meas[0:7]; est.qpos[qa] = x_meas[13:16]
            est.qvel[:] = 0; est.qvel[0:3] = x_meas[7:10]; est.qvel[3:6] = x_meas[10:13]; est.qvel[qv] = x_meas[16:19]
            mujoco.mj_forward(cmodel, est)
            if k % every == 0 and ctrl_name not in ("pid", "pid_tuned_noouter") and t >= 7.0:
                # same outer integral action as the MPC: integrate the gripper error, shift the base reference
                ee_int = np.clip(ee_int + (ee_r - est.site_xpos[ee_site]) * 0.02 * 1.5, -0.05, 0.05)
            b_ref = base_r + ee_int
            if ctrl_name.startswith("lqr"):
                thrusts = lqr(est.qpos.copy(), est.qvel.copy(), b_ref, base_v,
                              th_meas=x_meas[13:16] if k % every == 0 else None)
            else:
                thrusts, _ = geo(est, b_ref, v_d=base_v, dt=dt)
            if k % every == 0:
                R = quat_to_R(x_meas[3:7])
                th_ik = ik(P, R.T @ (ee_r - x_meas[0:3]), th_ik, lo, hi)   # arm IK from the measured base pose
            u = np.r_[thrusts, th_ik]
        elif k % every == 0:
            if ctrl_name.startswith("mpc"):
                if ctrl_name == "mpc" and t >= 7.0:      # integral action on the EE error once at the pad (offset-free tracking)
                    ee_now = np.asarray(mpc.wb.f_ee(x_meas)).ravel()
                    ee_int = np.clip(ee_int + (ee_r - ee_now) * 0.02 * 1.5, -0.05, 0.05)
                ee_tr, b_tr, v_tr = [], [], []
                for j in range(prm.N + 1):
                    e, b, bv, _ = ref(t + j * prm.dt)
                    ee_tr.append(e + ee_int); b_tr.append(b); v_tr.append(bv)
                mpc.set_trajectory(np.array(ee_tr), np.array(b_tr), np.array(v_tr), u_hover, th_ref=TH_NOM)
                if kf is not None:
                    mpc.set_disturbance(kf.d)            # Kalman disturbance estimate instead of the EE integrator
                u, _ = mpc(x_meas)
                if kf is not None:
                    kf.update(x_meas, u, np.zeros(3))
        data.ctrl[:n] = np.clip(u[:n], 0, cfg.thrust_max)
        th_r, _ = arm_ref.step(u[n:], dt)
        data.ctrl[n:] = th_r + data.qfrc_bias[qv] / cfg.servo_kp
        if realistic:
            wind += (-(wind - wind_mean) / 1.5) * dt + wind_sigma * np.sqrt(2 * dt / 1.5) * rng.normal(size=3) * np.array([1, 1, 0.3])
            data.xfrc_applied[base_id, :3] = wind
        data.mocap_pos[mark] = ee_r
        mujoco.mj_step(model, data)
        if k % 10 == 0:
            Rt = quat_to_R(data.qpos[3:7])
            for key, val in (("t", t), ("ee", data.site_xpos[ee_site].copy()), ("ee_ref", ee_r.copy()), ("base", data.qpos[0:3].copy()),
                             ("base_ref", base_r.copy()), ("tilt", np.degrees(np.arccos(np.clip(Rt[2, 2], -1, 1)))),
                             ("f", data.act[:n].copy()), ("th", data.qpos[qa].copy()), ("phase", ph)):
                log[key].append(val)
        if video and k % int(round(1 / (30 * dt))) == 0:
            cam.lookat[:] = 0.5 * (data.qpos[0:3] + data.site_xpos[ee_site]) + np.array([0.05, 0, 0.0])
            renderer.update_scene(data, camera=cam)
            frames.append((renderer.render(), t, ph, np.linalg.norm(data.site_xpos[ee_site] - ee_r), log["tilt"][-1]))
    L = {k: np.array(v) for k, v in log.items()}
    e = np.linalg.norm(L["ee"] - L["ee_ref"], axis=1)
    bdev = np.linalg.norm(L["base"] - L["base_ref"], axis=1)
    arm_sel = (L["t"] >= 9.0) & (L["t"] < 21.0)
    app_sel = (L["t"] >= 2.0) & (L["t"] < 9.0)
    hold_sel = L["t"] >= 21.5
    sat = np.mean((L["f"] <= 0.05) | (L["f"] >= cfg.thrust_max - 0.05)) * 100
    crashed = bool(L["tilt"].max() > 60.0 or L["base"][:, 2].min() < 0.2)
    res = dict(controller=ctrl_name, realistic=realistic, wind_mean_N=round(float(wind_mag), 3) if realistic else 0.0, crashed=crashed,
               ee_rms_arm_phase_mm=round(float(np.sqrt(np.mean(e[arm_sel] ** 2)) * 1e3), 1),
               ee_max_arm_phase_mm=round(float(e[arm_sel].max() * 1e3), 1),
               ee_rms_arm_phase_in_plane_mm=round(float(np.sqrt(np.mean(np.sum((L["ee"] - L["ee_ref"])[arm_sel][:, [0, 2]] ** 2, axis=1))) * 1e3), 1),
               ee_rms_arm_phase_sideways_mm=round(float(np.sqrt(np.mean((L["ee"] - L["ee_ref"])[arm_sel][:, 1] ** 2)) * 1e3), 1),
               ee_rms_approach_mm=round(float(np.sqrt(np.mean(e[app_sel] ** 2)) * 1e3), 1),
               ee_mean_hold_mm=round(float(e[hold_sel].mean() * 1e3), 1),
               base_dev_rms_arm_phase_mm=round(float(np.sqrt(np.mean(bdev[arm_sel] ** 2)) * 1e3), 1),
               max_tilt_deg=round(float(L["tilt"].max()), 1),
               rotor_at_limit_pct=round(float(sat), 2),
               arm_joint_travel_rad=[round(float(v), 2) for v in np.ptp(L["th"][L["t"] >= 7.0], axis=0)])
    if ctrl_name.startswith("lqr"):
        res["lqr_designs"] = lqr.n_relinearize
    if ctrl_name.startswith("mpc"):
        if kf is not None:
            res["d_hat_end_N"] = [round(float(v), 2) for v in kf.d]
        res["solve_ms_median"] = round(float(np.median(mpc.solve_times) * 1e3), 2)
        res["solver_ok_pct"] = round(100.0 * mpc.status_counts.get(0, 0) / sum(mpc.status_counts.values()), 1)
    return res, L, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    results, logs, frames = [], {}, {}
    for realistic in (False, True):
        for c in ("pid", "mpc"):
            r, L, fr = run(c, realistic, video=args.video and realistic)
            results.append(r); logs[(c, realistic)] = L; frames[c] = fr
            print(json.dumps(r))
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)

    fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for col, realistic in enumerate((False, True)):
        for c, sty in (("pid", "C1"), ("mpc", "C0")):
            L = logs[(c, realistic)]
            e = np.linalg.norm(L["ee"] - L["ee_ref"], axis=1)
            ax[0, col].semilogy(L["t"], np.maximum(e, 1e-4) * 1e3, color=sty, label=c.upper())
            ax[1, col].plot(L["t"], np.linalg.norm(L["base"] - L["base_ref"], axis=1) * 1e3, color=sty)
            ax[2, col].plot(L["t"], L["tilt"], color=sty)
        ax[0, col].set_title("ideal (perfect state, perfect model, no wind)" if not realistic else
                             "realistic (noise, 10 ms delay, +10% mass, 95% thrust, slower motors, wind)")
        for a in ax[:, col]:
            for x in (2, 7, 9, 21):
                a.axvline(x, color="0.85", lw=1)
            a.grid(alpha=0.3)
    ax[0, 0].set_ylabel("gripper error [mm]"); ax[1, 0].set_ylabel("base off its reference [mm]"); ax[2, 0].set_ylabel("tilt [deg]")
    ax[2, 0].set_xlabel("time [s]"); ax[2, 1].set_xlabel("time [s]"); ax[0, 0].legend()
    fig.suptitle("Whole-body MPC vs cascaded PID + arm IK · approach 2–7 s · reach 7–9 s · arm-only motion 9–21 s · hold")
    fig.tight_layout(); fig.savefig(f"{OUT}/compare.png", dpi=105)

    if args.video and frames.get("mpc") and frames.get("pid"):
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        fn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        out = []
        for (im_p, t, ph, ep, tp), (im_m, _, _, em, tm) in zip(frames["pid"], frames["mpc"]):
            tiles = []
            for im, lab, e_, tl in ((im_p, "Cascaded PID + arm IK", ep, tp), (im_m, "Whole-body MPC", em, tm)):
                I = Image.fromarray(im); d = ImageDraw.Draw(I, "RGBA")
                d.rectangle([0, 0, 720, 62], fill=(19, 23, 31, 215))
                d.text((12, 6), lab, font=fb, fill=(238, 236, 230))
                d.text((12, 34), f"{ph} · t {t:4.1f} s · gripper error {e_ * 1000:5.1f} mm · tilt {tl:4.1f}°", font=fn, fill=(242, 154, 74))
                tiles.append(np.asarray(I))
            out.append(np.concatenate([tiles[0], np.full((540, 8, 3), 245, np.uint8), tiles[1]], axis=1))
        imageio.mimwrite(f"{OUT}/compare_realistic.mp4", out, fps=30, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=8)
        print("video", f"{OUT}/compare_realistic.mp4")


if __name__ == "__main__":
    main()

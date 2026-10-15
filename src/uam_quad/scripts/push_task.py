"""Touch-and-push: the F450 + arm presses its fingertip on a vertical board with a set force.

Timeline
  0-2 s    hover
  2-7 s    approach: fly to a standoff in front of the board
  7-9 s    bring the fingertip to 1 cm in front of the board
  9-15 s   push with 2 N (admittance loop moves the fingertip reference until the force matches)
  15-20 s  push with 4 N
  20-24 s  release: pull the fingertip 5 cm back, hover

Contact: the board has a compliant contact (MuJoCo solref 45 ms, like a rubber fingertip); a stiff
10 ms contact made the fingertip bounce and, with noise and delay, flipped the drone.

Force control: admittance around the MPC. The fingertip reference along the board normal moves
by  d_dot = k_a (F_des - F_meas), so the reference goes into the board until the measured contact
force equals the setpoint. The fingertip force comes from MuJoCo's contact solver (a stand-in for a
fingertip force sensor), low-pass filtered, with noise in realistic mode.

Two MPC variants:
  aware : once in contact, the MPC is told the commanded contact force (world frame) as an external
          EE force, so it plans the base tilt needed to lean into the board
  plain : the MPC treats the push as an unknown disturbance

    python scripts/push_task.py                    # 4 runs (aware/plain x ideal/realistic), plot + json
    python scripts/push_task.py --video            # + video of the realistic force-aware run
"""
import argparse
import json
import os
import sys

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_controllers import START, TH_NOM, WIND_DIR, Sensors, min_jerk  # noqa: E402
from uam_quad import f450_stock, hover_trim, load_model, with_omx_arm  # noqa: E402
from uam_quad.paths import RUNS, BUILD  # noqa: E402
from uam_quad.controller import quat_to_R  # noqa: E402
from uam_quad.estimator import DisturbanceKF  # noqa: E402
from uam_quad.wbmpc import ArmReference, DisturbanceObserver, MPCParams, WholeBodyMPC, state_from_mujoco  # noqa: E402

OUT = os.path.join(RUNS, "push")
BOARD_X = 1.0                      # board front face (x) -- the fingertip pushes along +x
BOARD_Z = 1.30
TIP_R = 0.008                      # fingertip sphere radius
SCENE = f"""
    <body name="board_post" pos="{BOARD_X + 0.06} 0 0">
      <geom name="post" type="cylinder" fromto="0 0 0 0 0 {BOARD_Z - 0.1}" size="0.02" rgba="0.45 0.35 0.25 1"/>
    </body>
    <body name="board" pos="{BOARD_X + 0.006} 0 {BOARD_Z}">
      <geom name="board" type="box" size="0.006 0.12 0.10" rgba="0.3 0.65 0.35 1" contype="0" conaffinity="1"
            friction="0.6 0.005 0.0001" solref="0.045 1" solimp="0.9 0.95 0.002"/>
    </body>
    <body name="target_mark" mocap="true" pos="0 0 -1"><geom type="sphere" size="0.006" rgba="1 0.2 0.2 0.9" contype="0" conaffinity="0"/></body>"""
N_HAT = np.array([1.0, 0.0, 0.0])  # board normal, pointing into the board
TANGENTIAL_INTEGRAL = False
CONTACT_ON = 0.3   # [N] force that declares contact (leave: < 0.1 N for 0.1 s)


def force_setpoint(t):
    if 9.0 <= t < 15.0:
        return 2.0
    if 15.0 <= t < 20.0:
        return 4.0
    return 0.0


def contact_force(model, data, board_gid):
    """Total normal force the board exerts on the robot [N] (>= 0)."""
    total = 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if board_gid in (c.geom1, c.geom2):
            mujoco.mj_contactForce(model, data, i, f6)
            total += f6[0]
    return total


ALL_FACTORS = ("noise", "delay", "mismatch", "wind", "fnoise")


def run(aware, realistic, video=False, seed=0, wind_mag=0.335, factors=None, observer=False, estimator=False,
        est_axes="tangential"):
    """factors: subset of ALL_FACTORS to enable when realistic (default: all).
    est_axes: which estimated components reach the MPC -- 'freeze' (free-flight estimate held while in
    contact), 'tangential' (never along the board normal, which the admittance loop owns),
    'all_but_normal_in_contact' or 'vertical' (z only)."""
    rng = np.random.default_rng(seed)
    F = set(ALL_FACTORS if factors is None else factors) if realistic else set()
    nominal = with_omx_arm(f450_stock())
    plant_cfg = with_omx_arm(f450_stock(base_mass=nominal.base_mass * (1.10 if "mismatch" in F else 1.0),
                                        motor_tau=0.03 if "mismatch" in F else nominal.motor_tau))
    model, cfg = load_model(plant_cfg, extra_world=SCENE)
    cmodel, ccfg = load_model(nominal, extra_world=SCENE)
    if "mismatch" in F:
        model.actuator_gainprm[:4, 0] *= 0.95
    n, dt = cfg.n_rotors, model.opt.timestep
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    qv = [model.jnt_dofadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    ee_site, base_id = model.site("ee").id, model.body("base").id
    board_gid, mark = model.geom("board").id, model.body("target_mark").mocapid[0]

    trim = hover_trim(model, cfg, TH_NOM, pos=START)
    trim_nom = hover_trim(cmodel, ccfg, TH_NOM, pos=START)
    data = mujoco.MjData(model)
    data.qpos[:], data.ctrl[:] = trim.qpos, trim.ctrl
    data.act[:n] = trim.act
    mujoco.mj_forward(model, data)
    ee_off = data.site_xpos[ee_site] - START
    touch = np.array([BOARD_X - TIP_R, 0.0, BOARD_Z])     # fingertip centre when just touching
    pre = touch - np.array([0.01, 0, 0])                    # 1 cm in front of the board
    base_park = touch - np.array([0.06, 0, 0]) - ee_off     # arm does the last 6 cm

    prm = MPCParams(arm_wn=25.0, w_base=200.0, codegen_dir=os.path.join(BUILD, f"uam_wbmpc_{os.getpid()}"))
    mpc = WholeBodyMPC(cmodel, ccfg, prm, th_nominal=TH_NOM)
    u_hover = np.r_[trim_nom.ctrl[:n], TH_NOM]
    mpc.warm_start(state_from_mujoco(model, data, qa, qv), u_hover)
    obs = DisturbanceObserver(mpc.wb, dt=0.02) if observer else None
    kf = DisturbanceKF(mpc.wb, dt=0.02, motor_tau=ccfg.motor_tau) if estimator else None
    d_log = []
    d_frozen = np.zeros(3)
    sensors = Sensors("noise" in F or "delay" in F, rng, delay_steps=5 if "delay" in F else 0)
    if "noise" not in F:
        sensors.rng = type("NoNoise", (), {"normal": staticmethod(lambda m, s, n=None: np.zeros(n) if n else 0.0)})()
    arm_ref = ArmReference(TH_NOM, prm.arm_wn)
    wind_mean, wind = wind_mag * WIND_DIR, np.zeros(3)

    d = 0.0            # admittance offset of the fingertip reference along the normal [m]
    in_contact, below_since = False, None   # contact detection with hysteresis (enter > CONTACT_ON, leave < 0.1 N for 0.1 s)
    F_filt = 0.0
    k_adm = 0.006      # [m/s per N]
    ee_int = np.zeros(3)  # integral action on the fingertip error tangential to the board (y, z)
    every = int(round(0.02 / dt))
    u = u_hover.copy()

    def ee_ref_at(t):
        if t < 2.0:
            return START + ee_off, START
        if t < 7.0:
            b, _ = min_jerk(START, base_park, 5.0, t - 2.0)
            return b + ee_off, b
        if t < 9.0:
            e, _ = min_jerk(base_park + ee_off, pre, 2.0, t - 7.0)
            return e, base_park
        if t < 20.0:
            return pre + N_HAT * (0.01 + d) + ee_int, base_park  # admittance along the normal, integral along the board
        e, _ = min_jerk(pre + N_HAT * (0.01 + d), pre - N_HAT * 0.04, 1.5, t - 20.0)
        return e, base_park

    log = {k: [] for k in ("t", "F", "F_meas", "F_des", "d", "ee", "ee_ref", "base", "tilt", "f")}
    frames, renderer, cam = [], None, None
    if video:
        renderer = mujoco.Renderer(model, 540, 960)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 1.0, -120, -12
    for k in range(int(24.0 / dt)):
        t = k * dt
        F_true = contact_force(model, data, board_gid)
        F_s = F_true + (rng.normal(0, 0.1) if "fnoise" in F else 0.0)
        F_filt += (F_s - F_filt) * dt / 0.03
        F_des = force_setpoint(t)
        if k % every == 0:
            if 9.0 <= t < 20.0:
                d = float(np.clip(d + k_adm * (F_des - F_filt) * 0.02, -0.01, 0.05))
            x_meas = sensors(state_from_mujoco(model, data, qa, qv))
            if TANGENTIAL_INTEGRAL and t >= 7.0:   # tried for the realistic case; did not help, off by default
                ee_err = ee_ref_at(t)[0] - np.asarray(mpc.wb.f_ee(x_meas)).ravel()
                ee_int = np.clip(ee_int + np.array([0.0, 1.0, 1.0]) * ee_err * 0.02 * 1.5, -0.05, 0.05)
            ee_tr, b_tr = zip(*[ee_ref_at(t + j * prm.dt) for j in range(prm.N + 1)])
            mpc.set_trajectory(np.array(ee_tr), np.array(b_tr), np.zeros((prm.N + 1, 3)), u_hover, th_ref=TH_NOM)
            if not in_contact and F_filt > CONTACT_ON:
                in_contact, below_since = True, None
            elif in_contact and F_filt < 0.1:
                below_since = t if below_since is None else below_since
                if t - below_since > 0.1:
                    in_contact = False
            elif in_contact:
                below_since = None
            # feedforward the commanded force once in contact (smoother than the noisy measurement)
            f_ee = -N_HAT * F_des if (aware and in_contact) else np.zeros(3)
            mpc.set_contact_force(f_ee)
            if obs is not None:
                mpc.set_disturbance(obs.update(x_meas, u, f_ee))
            if kf is not None:
                d_hat = kf.d
                if est_axes == "freeze":
                    # estimate in free flight; hold the last free-flight value while touching (friction and
                    # contact transients are not base disturbances), never along the board normal
                    if in_contact:
                        d_hat = d_frozen
                    else:
                        d_frozen = d_hat - N_HAT * (N_HAT @ d_hat)
                        d_hat = d_frozen
                elif est_axes == "vertical":
                    d_hat = np.array([0.0, 0.0, d_hat[2]])
                elif est_axes == "tangential" or in_contact:   # the admittance loop owns the board normal
                    d_hat = d_hat - N_HAT * (N_HAT @ d_hat)
                mpc.set_disturbance(d_hat)
            u, _ = mpc(x_meas)
            if kf is not None:
                kf.update(x_meas, u, -N_HAT * max(F_filt, 0.0))   # measured contact force in the prediction
                d_log.append((t, *kf.d))
        else:
            sensors(state_from_mujoco(model, data, qa, qv))
        data.ctrl[:n] = np.clip(u[:n], 0, cfg.thrust_max)
        th_r, _ = arm_ref.step(u[n:], dt)
        data.ctrl[n:] = th_r + data.qfrc_bias[qv] / cfg.servo_kp
        if "wind" in F:
            wind += (-(wind - wind_mean) / 1.5) * dt + 1.5 * wind_mag * np.sqrt(2 * dt / 1.5) * rng.normal(size=3) * np.array([1, 1, 0.3])
            data.xfrc_applied[base_id, :3] = wind
        e_now, _ = ee_ref_at(t)
        data.mocap_pos[mark] = e_now
        mujoco.mj_step(model, data)
        if k % 10 == 0:
            R = quat_to_R(data.qpos[3:7])
            for key, val in (("t", t), ("F", F_true), ("F_meas", F_filt), ("F_des", F_des), ("d", d),
                             ("ee", data.site_xpos[ee_site].copy()), ("ee_ref", e_now.copy()), ("base", data.qpos[0:3].copy()),
                             ("tilt", np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), ("f", data.act[:n].copy())):
                log[key].append(val)
        if video and k % int(round(1 / (30 * dt))) == 0:
            cam.lookat[:] = 0.5 * (data.qpos[0:3] + data.site_xpos[ee_site])
            renderer.update_scene(data, camera=cam)
            frames.append((renderer.render(), t, F_true, F_des, log["tilt"][-1]))
    L = {k: np.array(v) for k, v in log.items()}
    w2 = (L["t"] >= 11.0) & (L["t"] < 15.0)
    w4 = (L["t"] >= 17.0) & (L["t"] < 20.0)
    push = (L["t"] >= 11.0) & (L["t"] < 20.0)
    res = dict(aware=aware, observer=observer, estimator=estimator, realistic=realistic,
               force_mean_2N=round(float(L["F"][w2].mean()), 2), force_rms_err_2N=round(float(np.sqrt(np.mean((L["F"][w2] - 2.0) ** 2))), 2),
               force_mean_4N=round(float(L["F"][w4].mean()), 2), force_rms_err_4N=round(float(np.sqrt(np.mean((L["F"][w4] - 4.0) ** 2))), 2),
               contact_lost_pct=round(float(np.mean(L["F"][push] < 0.05) * 100), 1),
               base_x_drift_mm=round(float((L["base"][push, 0] - L["base"][L["t"] < 9.0][-1, 0]).min() * 1e3), 1),
               max_tilt_deg=round(float(L["tilt"].max()), 1),
               final_admittance_offset_mm=round(float(L["d"][w4][-1] * 1e3), 1),
               solver_ok_pct=round(100.0 * mpc.status_counts.get(0, 0) / sum(mpc.status_counts.values()), 1))
    if d_log:
        D = np.array(d_log)
        late = D[:, 0] >= 5.0
        res["d_hat_mean_N"] = [round(float(v), 2) for v in D[late, 1:].mean(axis=0)]
        L["d_hat"] = D
    return res, L, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out, logs, frames = [], {}, None
    for realistic in (False, True):
        for aware in (True, False):
            r, L, fr = run(aware, realistic, video=a.video and aware and realistic)
            out.append(r); logs[(aware, realistic)] = L
            if fr:
                frames = fr
            print(json.dumps(r), flush=True)
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1)
    fig, ax = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for col, realistic in enumerate((False, True)):
        for aware, c in ((True, "C0"), (False, "C1")):
            L = logs[(aware, realistic)]
            lab = "force-aware MPC" if aware else "plain MPC"
            ax[0, col].plot(L["t"], L["F"], color=c, label=lab, lw=1)
            ax[1, col].plot(L["t"], (L["base"][:, 0] - L["base"][0, 0]) * 1, color=c)
            ax[2, col].plot(L["t"], L["tilt"], color=c)
        ax[0, col].plot(L["t"], L["F_des"], "k--", lw=1, label="setpoint")
        ax[0, col].set_title("ideal" if not realistic else "realistic (noise, delay, +10% mass, 95% thrust, slow motors, wind, 0.1 N force noise)")
        for a_ in ax[:, col]:
            for x in (2, 7, 9, 15, 20):
                a_.axvline(x, color="0.85", lw=1)
            a_.grid(alpha=0.3)
    ax[0, 0].set_ylabel("contact force [N]"); ax[1, 0].set_ylabel("base x from start [m]"); ax[2, 0].set_ylabel("tilt [deg]")
    ax[0, 0].legend(); ax[2, 0].set_xlabel("time [s]"); ax[2, 1].set_xlabel("time [s]")
    fig.suptitle("Touch and push: fingertip on a vertical board · 2 N from 9 s · 4 N from 15 s · release at 20 s")
    fig.tight_layout(); fig.savefig(f"{OUT}/push.png", dpi=105)
    if frames:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        fn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
        vid = []
        for img, t, F, Fd, tilt in frames:
            I = Image.fromarray(img); dr = ImageDraw.Draw(I, "RGBA")
            dr.rectangle([0, 0, 960, 66], fill=(19, 23, 31, 215))
            dr.text((12, 6), "Touch and push · force-aware whole-body MPC · realistic conditions", font=fb, fill=(238, 236, 230))
            dr.text((12, 38), f"t {t:4.1f} s   contact force {F:4.2f} N (setpoint {Fd:.0f} N)   tilt {tilt:4.1f}°", font=fn, fill=(242, 154, 74))
            bw = int(min(F, 6.0) / 6.0 * 300)
            dr.rectangle([640, 470, 940, 500], outline=(200, 200, 200), width=2)
            dr.rectangle([640, 470, 640 + bw, 500], fill=(242, 154, 74))
            if Fd > 0:
                xs = 640 + int(Fd / 6.0 * 300); dr.line([xs, 462, xs, 508], fill=(255, 255, 255), width=3)
            dr.text((640, 440), "contact force (white = setpoint)", font=fn, fill=(238, 236, 230))
            vid.append(np.asarray(I))
        imageio.mimwrite(f"{OUT}/push_realistic.mp4", vid, fps=30, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=4)
        print("video", f"{OUT}/push_realistic.mp4")


if __name__ == "__main__":
    main()

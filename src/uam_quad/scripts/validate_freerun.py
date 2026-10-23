"""Free-run (multi-step) check of the MPC prediction model against MuJoCo, with a fingertip force.

Both start from the same hover state at the nominal arm pose. Both get the same input sequence for
T = 1.0 s (the MPC horizon):
  - rotor thrust commands: hover trim + smooth random perturbation (sum of 3 sines, up to 0.3 N);
  - arm commands: nominal + smooth random perturbation (up to 0.3 rad);
  - a constant fingertip force f_ext = -F n_hat (n_hat = +x), F in {0, 2, 4} N.
MuJoCo applies f_ext at the ee site (mj_applyFT), passes thrust through its 20 ms motor lag, and
drives the arm servos with the arm reference model + gravity feedforward (as in push_task.py). The
model (wbmpc.WholeBodyModel.f_dyn, thrusts mode) is integrated open-loop with RK4 at 2 ms, with no
feedback from MuJoCo. Reported: hub position, fingertip position and attitude error at 0.2, 0.5 and 1.0 s.
Two variants: "cmd" feeds the model the commanded thrusts (what the MPC predicts with: no motor lag);
"act" feeds it the thrusts MuJoCo actually applies after its motor lag, isolating the rigid-body + arm
equations from the unmodelled lag. Arm servo tracking error is in both. "act_ff" is "act" with the
contact torque J_ee^T f_ext added to the servo feedforward, which removes most of the servo deflection
the fingertip force causes (the model assumes the arm follows its reference regardless of load).

    python scripts/validate_freerun.py            # 10 seeds x 3 force levels -> runs/uam_quad/validate_freerun.json
"""
import json
import os

import mujoco
import numpy as np

from uam_quad import f450_stock, hover_trim, load_model, with_omx_arm
from uam_quad.paths import RUNS
from uam_quad.wbmpc import ArmReference, MPCParams, WholeBodyModel, state_from_mujoco

TH_NOM = np.array([-0.6, 1.0, 0.4])
N_HAT = np.array([1.0, 0.0, 0.0])
T_END, DT = 1.0, 0.002


def quat_angle(q1, q2):
    return 2 * np.arccos(np.clip(abs(float(np.dot(q1, q2))), -1.0, 1.0))


def run(seed, F, thrust_input):
    contact_ff = thrust_input == "act_ff"
    cfg = with_omx_arm(f450_stock())
    model, cfg = load_model(cfg)
    wb = WholeBodyModel(model, cfg, MPCParams(arm_wn=25.0))
    n = cfg.n_rotors
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    qv = [model.jnt_dofadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    ee_site, ee_body = model.site("ee").id, model.body("ee").id
    tr = hover_trim(model, cfg, TH_NOM, pos=(0, 0, 3.0))
    d = mujoco.MjData(model)
    d.qpos[:], d.ctrl[:] = tr.qpos, tr.ctrl
    d.act[:n] = tr.act
    mujoco.mj_forward(model, d)
    x = state_from_mujoco(model, d, qa, qv)
    f_ext = -F * N_HAT
    p = np.r_[f_ext, np.zeros(3)]
    jac = np.zeros((3, model.nv))
    arm = ArmReference(TH_NOM, wb.prm.arm_wn)
    fdyn = lambda x_, u_: np.asarray(wb.f_dyn(x_, u_, p)).ravel()  # noqa: E731
    out = {}
    # per-seed perturbation shapes: 3 sines per channel, 0.5-3 Hz, peak <= 0.3 N (rotors) / 0.3 rad (arm)
    fs = np.random.default_rng([seed, 1]); as_ = np.random.default_rng([seed, 2])
    ff, fph, fa = fs.uniform(0.5, 3, (n, 3)), fs.uniform(0, 2 * np.pi, (n, 3)), fs.uniform(0.3, 1, (n, 3))
    af, aph, aa = as_.uniform(0.5, 3, (3, 3)), as_.uniform(0, 2 * np.pi, (3, 3)), as_.uniform(0.3, 1, (3, 3))
    fa = 0.3 * fa / fa.sum(1, keepdims=True); aa = 0.3 * aa / aa.sum(1, keepdims=True)
    for k in range(int(round(T_END / DT))):
        t = k * DT
        f_cmd = tr.ctrl[:n] + (fa * np.sin(2 * np.pi * ff * t + fph)).sum(1)
        th_cmd = TH_NOM + (aa * np.sin(2 * np.pi * af * t + aph)).sum(1)
        u = np.r_[f_cmd if thrust_input == "cmd" else d.act[:n].copy(), th_cmd]
        # MuJoCo
        d.ctrl[:n] = f_cmd
        th_r, _ = arm.step(th_cmd, DT)
        tau_contact = np.zeros(3)
        if contact_ff:
            mujoco.mj_jacSite(model, d, jac, None, ee_site)
            tau_contact = jac[:, qv].T @ f_ext
        d.ctrl[n:] = th_r + (d.qfrc_bias[qv] - tau_contact) / cfg.servo_kp
        d.xfrc_applied[:] = 0
        mujoco.mj_applyFT(model, d, f_ext, np.zeros(3), d.site_xpos[ee_site].copy(), ee_body, d.qfrc_applied)
        mujoco.mj_step(model, d)
        d.qfrc_applied[:] = 0
        # model, RK4
        k1 = fdyn(x, u); k2 = fdyn(x + DT / 2 * k1, u); k3 = fdyn(x + DT / 2 * k2, u); k4 = fdyn(x + DT * k3, u)
        x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x[3:7] /= np.linalg.norm(x[3:7])
        tt = round((k + 1) * DT, 6)
        if tt in (0.2, 0.5, 1.0):
            out[tt] = dict(
                hub_mm=float(np.linalg.norm(x[0:3] - d.qpos[0:3]) * 1e3),
                ee_mm=float(np.linalg.norm(np.asarray(wb.f_ee(x)).ravel() - d.site_xpos[ee_site]) * 1e3),
                att_deg=float(np.degrees(quat_angle(x[3:7], d.qpos[3:7]))),
                hub_travel_mm=float(np.linalg.norm(d.qpos[0:3] - tr.qpos[0:3]) * 1e3))
    return out


def main():
    rows = []
    for mode in ("cmd", "act", "act_ff"):
        for F in (0.0, 2.0, 4.0):
            for seed in range(10):
                r = run(seed, F, mode)
                rows.append(dict(mode=mode, F=F, seed=seed, **{f"{k}@{t}": v for t, d_ in r.items() for k, v in d_.items()}))
    os.makedirs(RUNS, exist_ok=True)
    json.dump(rows, open(os.path.join(RUNS, "validate_freerun.json"), "w"), indent=1)
    keys = [k for k in rows[0] if "@" in k]
    for mode, F in [(m, F) for m in ("cmd", "act", "act_ff") for F in (0.0, 2.0, 4.0)]:
        R = [r for r in rows if r["F"] == F and r["mode"] == mode]
        print(f"{mode} F={F:.0f} N  " + "  ".join(f"{k}: med {np.median([r[k] for r in R]):.2f} max {np.max([r[k] for r in R]):.2f}" for k in keys))


if __name__ == "__main__":
    main()

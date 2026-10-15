"""Hand-derived equations of motion for the multirotor + 3-DoF pitch arm (derivation: docs/eom.md).

The base equations are Newton-Euler for the whole system about the base origin O, written in the
body frame, with the arm motion (theta, theta_dot, theta_ddot) treated as given:

  translation:  m (a_O + w_dot x c + w x (w x c) + 2 w x c_dot + c_ddot) = F_B - m g R^T e3
  rotation:     J_O w_dot + J_O_dot w + w x (J_O w) + h_dot + w x h + m c x a_O
                    = M_B + m c x (-g R^T e3)

  a_O     body-frame acceleration of O (world accel p_ddot = R a_O)
  c       whole-system CoM relative to O (body frame), depends on theta
  J_O     whole-system inertia about O (body frame), depends on theta
  h       arm angular momentum about O from joint motion only (body frame)
  F_B,M_B rotor wrench about O (body frame) = allocation_matrix @ thrusts

Everything arm-related is computed from our own forward kinematics and the per-body masses and
inertias, not from MuJoCo's kinematics, so `check_against_mujoco` is an independent test.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .model import UAMConfig

E2, E3 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def Ry(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # rotates +x toward -z for phi > 0


class ArmBodyParams:
    """Mass properties pulled once from the compiled model (masses, CoM offsets, inertias)."""

    def __init__(self, model, cfg: UAMConfig):
        def props(name):
            b = model.body(name).id
            Rq = np.zeros(9)
            mujoco.mju_quat2Mat(Rq, model.body_iquat[b])
            Rq = Rq.reshape(3, 3)
            return model.body_mass[b], model.body_ipos[b].copy(), Rq @ np.diag(model.body_inertia[b]) @ Rq.T

        self.base = props("base")
        self.links = [props(n) for n in ("arm_l1", "arm_l2", "arm_l3", "ee")]
        self.mount = np.array(cfg.arm_mount, dtype=float)
        self.L = np.array(cfg.link_lengths, dtype=float)
        self.g = -model.opt.gravity[2]


def arm_frames(P: ArmBodyParams, th):
    """Origin and rotation (in the base frame) of arm_l1, arm_l2, arm_l3, ee, plus pitch angles."""
    phi = np.cumsum(th)
    phis = [phi[0], phi[1], phi[2], phi[2]]
    origins, o = [], P.mount.copy()
    for k in range(4):
        origins.append(o.copy())
        if k < 3:
            o = o + Ry(phi[k]) @ np.array([P.L[k], 0.0, 0.0])
    return origins, phis


def mass_terms(P: ArmBodyParams, th, thd):
    """m, c, J_O and h at (theta, theta_dot), all in the base frame."""
    mb, rb, Ib = P.base
    m = mb + sum(l[0] for l in P.links)
    origins, phis = arm_frames(P, th)
    phid = np.cumsum(thd)
    phids = [phid[0], phid[1], phid[2], phid[2]]
    mc = mb * rb
    J = Ib + mb * (rb @ rb * np.eye(3) - np.outer(rb, rb))
    h = np.zeros(3)
    for (mk, ipos, Ik_local), o, ph, phd in zip(P.links, origins, phis, phids):
        Rk = Ry(ph)
        r = o + Rk @ ipos
        Ik = Rk @ Ik_local @ Rk.T
        mc += mk * r
        J += Ik + mk * (r @ r * np.eye(3) - np.outer(r, r))
    # h: angular momentum about O due to joint motion only = sum m r x r_dot + I (phi_dot e2)
    eps = 1e-5
    for k, ((mk, ipos, Ik_local), ph, phd) in enumerate(zip(P.links, phis, phids)):
        r_of = lambda t: (arm_frames(P, t)[0][k] + Ry(arm_frames(P, t)[1][k]) @ ipos)  # noqa: E731
        rdot = (r_of(th + eps * thd) - r_of(th - eps * thd)) / (2 * eps)
        Rk = Ry(ph)
        h += mk * np.cross(r_of(th), rdot) + Rk @ Ik_local @ Rk.T @ (phd * E2)
    return m, mc / m, J, h


def base_accelerations(P: ArmBodyParams, R, w, th, thd, thdd, F_B, M_B, drop=()):
    """Solve the two Newton-Euler equations for (p_ddot world, w_dot body)."""
    m, c, J, h = mass_terms(P, th, thd)
    eps = 1e-4

    def ddt(f):  # time derivative along the arm trajectory: df/dth * thd + df/dthd * thdd
        return (f(th + eps * thd, thd + eps * thdd) - f(th - eps * thd, thd - eps * thdd)) / (2 * eps)

    cdot = ddt(lambda t, td: mass_terms(P, t, td)[1])
    cddot = ddt(lambda t, td: (mass_terms(P, t + 1e-4 * td, td)[1] - mass_terms(P, t - 1e-4 * td, td)[1]) / 2e-4)
    Jdot = ddt(lambda t, td: mass_terms(P, t, td)[2])
    hdot = ddt(lambda t, td: mass_terms(P, t, td)[3])

    gB = -P.g * (R.T @ E3)  # gravity acceleration in the body frame
    # `drop` removes named terms, to see what each one is worth (docs/eom.md, term table)
    if "arm_reaction" in drop:
        hdot, h = np.zeros(3), np.zeros(3)
    if "com_motion" in drop:
        cdot, cddot = np.zeros(3), np.zeros(3)
    if "inertia_change" in drop:
        Jdot = np.zeros((3, 3))
    if "com_offset" in drop:
        c = np.zeros(3)
    A = np.block([[m * np.eye(3), -m * hat(c)], [m * hat(c), J]])
    rhs_t = F_B + m * gB - m * (np.cross(w, np.cross(w, c)) + 2 * np.cross(w, cdot) + cddot)
    rhs_r = M_B + m * np.cross(c, gB) - Jdot @ w - np.cross(w, J @ w) - hdot - np.cross(w, h)
    sol = np.linalg.solve(A, np.r_[rhs_t, rhs_r])
    return R @ sol[:3], sol[3:], dict(m=m, c=c, J=J, h=h, cdot=cdot, hdot=hdot)


def check_against_mujoco(model, cfg: UAMConfig, rng=None, trials=20, drop=()):
    """Random states and inputs: compare our base accelerations with MuJoCo's. Returns max errors."""
    rng = rng or np.random.default_rng(0)
    P = ArmBodyParams(model, cfg)
    A_alloc = cfg.allocation_matrix()
    n = cfg.n_rotors
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    data = mujoco.MjData(model)
    worst_lin = worst_ang = 0.0
    for _ in range(trials):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        data.qpos[:] = 0
        data.qpos[0:3] = [0, 0, 2.0]
        data.qpos[3:7] = q
        data.qpos[qa] = [rng.uniform(lo + 0.1, hi - 0.1) for lo, hi in cfg.joint_ranges]
        data.qvel[:] = rng.normal(scale=[1, 1, 1, 2, 2, 2, 3, 3, 3])
        data.act[:n] = rng.uniform(0, cfg.thrust_max, n)
        data.ctrl[:n] = data.act[:n]
        data.ctrl[n:] = data.qpos[qa] + rng.normal(scale=0.3, size=3)
        mujoco.mj_forward(model, data)
        assert data.ncon == 0
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[3:7])
        R = R.reshape(3, 3)
        wrench = A_alloc @ data.act[:n]
        F_B, M_B = np.array([0, 0, wrench[0]]), wrench[1:]
        pdd, wd, _ = base_accelerations(P, R, data.qvel[3:6], data.qpos[qa], data.qvel[6:9], data.qacc[6:9], F_B, M_B, drop)
        worst_lin = max(worst_lin, np.abs(pdd - data.qacc[0:3]).max())
        worst_ang = max(worst_ang, np.abs(wd - data.qacc[3:6]).max())
    return worst_lin, worst_ang

"""Whole-body MPC for an underactuated multirotor with a 3-DoF pitch arm (CasADi + ACADOS).

Prediction model = the coupled Newton-Euler equations of docs/eom.md, written symbolically:

  state  x = [p (3, world), q (4, wxyz body->world), v (3, world), w (3, body), th (3), thd (3)]  -> 19
  input  u = [f (n rotor thrusts), th_cmd (3)]                                                   -> n+3

Arm: the joints follow a critically damped reference model
    th_dd = wa^2 (th_cmd - th) - 2 wa th_d
which the (much stiffer) joint servos in the simulator track. Because the model knows th_dd, the
arm's reaction on the base (h_dot, c_ddot, J_dot terms) is part of the prediction -- the thing the
baseline geometric controller can't anticipate.

Cost (NONLINEAR_LS): end-effector position (main objective), weak base-position anchor,
upright attitude, small velocities and rates, arm posture, inputs near hover.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import casadi as ca
import mujoco
import numpy as np

from .eom import ArmBodyParams
from .paths import BUILD
from .model import UAMConfig

NX = 19


# ---------------------------------------------------------------------------------------------
# Symbolic model
# ---------------------------------------------------------------------------------------------
def _Ry(phi):
    c, s = ca.cos(phi), ca.sin(phi)
    return ca.vertcat(ca.horzcat(c, 0, s), ca.horzcat(0, 1, 0), ca.horzcat(-s, 0, c))


def _quat_to_R(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _hat(w):
    return ca.vertcat(ca.horzcat(0, -w[2], w[1]), ca.horzcat(w[2], 0, -w[0]), ca.horzcat(-w[1], w[0], 0))


@dataclass
class MPCParams:
    arm_wn: float = 12.0          # arm reference-model bandwidth [rad/s] (critically damped)
    dt: float = 0.04              # shooting interval [s]
    N: int = 25                   # horizon steps (1.0 s)
    # weights
    w_ee: float = 400.0
    w_base: float = 2.0
    w_att: tuple = (40.0, 40.0, 20.0)
    w_v: float = 4.0
    w_w: float = 1.0
    w_th: float = 0.02
    w_thd: float = 0.002
    w_f: float = 0.02
    w_thcmd: float = 0.02
    terminal_scale: float = 5.0
    max_tilt: float = 0.6         # |q_x|, |q_y| bound (≈ 69 deg tilt) as a sanity box
    coupling: bool = True         # False = ablation: drop arm-motion terms (h, h_dot, c_dot, c_ddot, J_dot)
    # Input mode. "thrusts": u = [rotor thrusts f_1..n, q_cmd] (MPC is the whole flight controller).
    # "rates": u = [collective thrust T, body-rate setpoint w_cmd (3, FLU), q_cmd] -- what PX4 offboard
    # takes (vehicle_rates_setpoint). PX4's rate loop is modelled as a torque source (see WholeBodyModel).
    input_mode: str = "thrusts"
    tau_rate: float = 0.05        # closed-loop time constant of PX4's rate controller [s]
    w_rate_cmd: tuple = (0.5, 0.5, 0.5)   # cost on the body-rate setpoint (rates mode)
    max_rate: tuple = (3.0, 3.0, 1.5)     # |w_cmd| bounds [rad/s] (rates mode)
    w_thcmd_rel: float = 0.5      # rates mode: cost on q_cmd - q (commanded arm acceleration) instead of q_cmd - q_ref;
                                  # damps a 12.5 Hz arm-command limit cycle seen with the weak absolute cost
    codegen_dir: str = field(default_factory=lambda: os.path.join(BUILD, "uam_wbmpc"))


class WholeBodyModel:
    """Symbolic coupled dynamics built from the compiled MuJoCo model's mass properties."""

    def __init__(self, model, cfg: UAMConfig, prm: MPCParams):
        self.cfg, self.prm = cfg, prm
        P = ArmBodyParams(model, cfg)
        self.P = P
        n = cfg.n_rotors
        self.n = n
        A = cfg.allocation_matrix()

        rates = prm.input_mode == "rates"
        self.rates = rates
        self.nu_flight = 4 if rates else n         # [T, w_cmd] or rotor thrusts
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", self.nu_flight + 3)
        f_ext = ca.SX.sym("f_ext", 3)          # external force on the end effector, world frame [N] (contact)
        f_dist = ca.SX.sym("f_dist", 3)        # estimated disturbance force on the base, world frame [N] (observer)
        thdd_in = ca.SX.sym("thdd", 3)
        p, q, v, w = x[0:3], x[3:7], x[7:10], x[10:13]
        th, thd = x[13:16], x[16:19]
        f, thcmd = u[: self.nu_flight], u[self.nu_flight:]

        # --- arm kinematics and mass terms (base frame) -------------------------------------
        phis = [th[0], th[0] + th[1], th[0] + th[1] + th[2]]
        origins, o = [], ca.DM(P.mount)
        for k in range(3):
            origins.append(o)
            o = o + _Ry(phis[k]) @ ca.DM([P.L[k], 0.0, 0.0])
        origins.append(o)                        # ee body origin
        phis.append(phis[2])
        ee_B = o

        mb, rb, Ib = P.base
        m = mb + sum(l[0] for l in P.links)
        mc = ca.DM(mb * rb)
        J = ca.DM(Ib + mb * (rb @ rb * np.eye(3) - np.outer(rb, rb)))
        rs, Is = [], []
        for (mk, ipos, Ik_local), ok, ph in zip(P.links, origins, phis):
            Rk = _Ry(ph)
            r = ok + Rk @ ca.DM(ipos)
            Ik = Rk @ ca.DM(Ik_local) @ Rk.T
            rs.append(r)
            Is.append(Ik)
            mc = mc + mk * r
            J = J + Ik + mk * (ca.dot(r, r) * ca.DM.eye(3) - r @ r.T)
        c = mc / m
        phid = [thd[0], thd[0] + thd[1], thd[0] + thd[1] + thd[2], thd[0] + thd[1] + thd[2]]
        h = ca.DM.zeros(3)
        for (mk, _, _), r, Ik, phd in zip(P.links, rs, Is, phid):
            rdot = ca.jacobian(r, th) @ thd
            h = h + mk * ca.cross(r, rdot) + Ik @ ca.vertcat(0, phd, 0)

        # time derivatives along the arm motion (th_dd given)
        cdot = ca.jacobian(c, th) @ thd
        cddot = ca.jacobian(cdot, th) @ thd + ca.jacobian(c, th) @ thdd_in
        Jdot = ca.reshape(ca.jacobian(ca.reshape(J, 9, 1), th) @ thd, 3, 3)
        hdot = ca.jacobian(h, th) @ thd + ca.jacobian(h, thd) @ thdd_in

        if not prm.coupling:  # ablation: the model knows where the arm is, not how it moves
            h, hdot, cdot, cddot, Jdot = ca.DM.zeros(3), ca.DM.zeros(3), ca.DM.zeros(3), ca.DM.zeros(3), ca.DM.zeros(3, 3)

        # --- base Newton-Euler about O (body frame) -----------------------------------------
        R = _quat_to_R(q)
        gB = -P.g * (R.T @ ca.DM([0, 0, 1]))
        f_ext_B = R.T @ f_ext
        rhs_corr = m * (ca.cross(w, ca.cross(w, c)) + 2 * ca.cross(w, cdot) + cddot)
        if not rates:
            wrench = ca.DM(A) @ f
            F_B = ca.vertcat(0, 0, wrench[0]) + f_ext_B + R.T @ f_dist   # contact + estimated disturbance ...
            M_B = wrench[1:4] + ca.cross(ee_B, f_ext_B)            # ... at the end effector, so it also twists the base
            Mmat = ca.vertcat(ca.horzcat(m * ca.DM.eye(3), -m * _hat(c)), ca.horzcat(m * _hat(c), J))
            rhs_t = F_B + m * gB - rhs_corr
            rhs_r = M_B + m * ca.cross(c, gB) - Jdot @ w - ca.cross(w, J @ w) - hdot - ca.cross(w, h)
            acc = ca.solve(Mmat, ca.vertcat(rhs_t, rhs_r))
            pdd = R @ acc[0:3]
            wd = acc[3:6]
            self.base_acc = ca.Function("base_acc", [x, f, thdd_in, f_ext, f_dist], [pdd, wd])  # for validation
        else:
            # PX4's rate loop as a torque source inside the full coupled equations:
            #   M_rate = J (w_cmd - w) / tau_rate + w x J w - m c x g_B
            # The last term is the steady gravity moment of the arm-shifted CoM, which the rate loop's
            # integrator cancels. The arm's motion-reaction terms (h_dot, J_dot w, c_ddot) are NOT
            # cancelled, so the MPC still predicts how moving the arm disturbs the base.
            T, w_cmd = f[0], f[1:4]
            M_rate = J @ (w_cmd - w) / prm.tau_rate + ca.cross(w, J @ w) - m * ca.cross(c, gB)
            F_B = ca.vertcat(0, 0, T) + f_ext_B + R.T @ f_dist
            M_B = M_rate + ca.cross(ee_B, f_ext_B)
            Mmat = ca.vertcat(ca.horzcat(m * ca.DM.eye(3), -m * _hat(c)), ca.horzcat(m * _hat(c), J))
            rhs_t = F_B + m * gB - rhs_corr
            rhs_r = M_B + m * ca.cross(c, gB) - Jdot @ w - ca.cross(w, J @ w) - hdot - ca.cross(w, h)
            acc = ca.solve(Mmat, ca.vertcat(rhs_t, rhs_r))
            pdd = R @ acc[0:3]
            wd = acc[3:6]

        wa = prm.arm_wn
        thdd = wa ** 2 * (thcmd - th) - 2 * wa * thd
        qdot = 0.5 * ca.vertcat(-ca.dot(q[1:4], w),
                                q[0] * w + ca.cross(q[1:4], w))
        xdot = ca.vertcat(v, qdot, ca.substitute(pdd, thdd_in, thdd), ca.substitute(wd, thdd_in, thdd), thd, thdd)

        self.x, self.u, self.xdot, self.p = x, u, xdot, ca.vertcat(f_ext, f_dist)
        self.thrust_max_total = float(n * cfg.thrust_max)
        self.f_dyn = ca.Function("f_dyn", [x, u, self.p], [xdot])
        self.ee_world = p + R @ ee_B
        self.f_ee = ca.Function("ee_world", [x], [self.ee_world])
        self.mass = float(m)


# ---------------------------------------------------------------------------------------------
# OCP
# ---------------------------------------------------------------------------------------------
class WholeBodyMPC:
    def __init__(self, model, cfg: UAMConfig, prm: MPCParams | None = None, th_nominal=(-0.6, 1.0, 0.4)):
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        self.prm = prm = prm or MPCParams()
        self.cfg = cfg
        self.wb = wb = WholeBodyModel(model, cfg, prm)
        n = cfg.n_rotors
        self.n = n
        self.th_nom = np.asarray(th_nominal, dtype=float)

        am = AcadosModel()
        am.name = f"uam_wbmpc_{n}r" + ("" if prm.coupling else "_nocpl") + ("_rates" if wb.rates else "")
        self.nf = nf = wb.nu_flight
        am.x, am.u, am.p = wb.x, wb.u, wb.p
        xd = ca.SX.sym("xdot", NX)
        am.xdot = xd
        am.f_expl_expr = wb.xdot
        am.f_impl_expr = xd - wb.xdot

        x, u = wb.x, wb.u
        self.thcmd_rel = wb.rates
        u_cost = ca.vertcat(u[:nf], u[nf:] - x[13:16]) if wb.rates else u
        y = ca.vertcat(wb.ee_world, x[0:3], x[4:7], x[7:10], x[10:13], x[13:16], x[16:19], u_cost)
        y_e = ca.vertcat(wb.ee_world, x[0:3], x[4:7], x[7:10], x[10:13], x[13:16], x[16:19])
        am.cost_y_expr, am.cost_y_expr_e = y, y_e
        self.ny, self.ny_e = y.shape[0], y_e.shape[0]

        ocp = AcadosOcp()
        ocp.model = am
        ocp.solver_options.N_horizon = prm.N
        ocp.solver_options.tf = prm.N * prm.dt
        w_flight = [prm.w_f / n] + list(prm.w_rate_cmd) if wb.rates else [prm.w_f] * n
        w_arm_cmd = prm.w_thcmd_rel if wb.rates else prm.w_thcmd
        W = np.diag(np.r_[[prm.w_ee] * 3, [prm.w_base] * 3, prm.w_att, [prm.w_v] * 3, [prm.w_w] * 3,
                          [prm.w_th] * 3, [prm.w_thd] * 3, w_flight, [w_arm_cmd] * 3])
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = W
        ocp.cost.W_e = prm.terminal_scale * W[: self.ny_e, : self.ny_e]
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(self.ny_e)

        lo = [r[0] for r in cfg.joint_ranges]
        hi = [r[1] for r in cfg.joint_ranges]
        if wb.rates:
            ocp.constraints.lbu = np.r_[0.0, -np.asarray(prm.max_rate), lo]
            ocp.constraints.ubu = np.r_[wb.thrust_max_total, prm.max_rate, hi]
        else:
            ocp.constraints.lbu = np.r_[np.zeros(n), lo]
            ocp.constraints.ubu = np.r_[np.full(n, cfg.thrust_max), hi]
        ocp.constraints.idxbu = np.arange(nf + 3)
        ocp.constraints.lbx = np.r_[[-prm.max_tilt] * 2, lo]
        ocp.constraints.ubx = np.r_[[prm.max_tilt] * 2, hi]
        ocp.constraints.idxbx = np.array([4, 5, 13, 14, 15])
        ocp.constraints.x0 = np.r_[0, 0, 1, 1, 0, 0, 0, np.zeros(6), self.th_nom, np.zeros(3)]
        ocp.parameter_values = np.zeros(6)
        self.f_ee_param, self.f_dist_param = np.zeros(3), np.zeros(3)

        so = ocp.solver_options
        so.integrator_type = "ERK"
        so.sim_method_num_stages = 4
        so.sim_method_num_steps = 2
        so.nlp_solver_type = "SQP_RTI"
        so.hessian_approx = "GAUSS_NEWTON"
        so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        so.qp_solver_iter_max = 50
        so.print_level = 0

        os.makedirs(prm.codegen_dir, exist_ok=True)
        ocp.code_export_directory = os.path.join(prm.codegen_dir, "c_generated_code")
        json_file = os.path.join(prm.codegen_dir, f"{am.name}.json")
        cwd = os.getcwd()
        os.chdir(prm.codegen_dir)  # acados builds relative to cwd
        try:
            self.solver = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
        finally:
            os.chdir(cwd)

        self.u_hover = None
        self.solve_times = []
        self.status_counts = {}

    # -- references ---------------------------------------------------------------------------
    def set_target(self, ee_target, base_anchor, u_hover, th_ref=None, yaw=0.0):
        th_ref = self.th_nom if th_ref is None else np.asarray(th_ref, dtype=float)
        qv = np.array([0.0, 0.0, np.sin(yaw / 2)])
        yref = np.r_[ee_target, base_anchor, qv, np.zeros(3), np.zeros(3), th_ref, np.zeros(3), u_hover[: self.nf],
                     np.zeros(3) if self.thcmd_rel else th_ref]
        for k in range(self.prm.N):
            self.solver.set(k, "yref", yref)
        self.solver.set(self.prm.N, "yref", yref[: self.ny_e])
        self.u_hover = np.asarray(u_hover, dtype=float)

    def set_trajectory(self, ee_traj, base_traj, base_vel, u_hover, th_ref=None, yaw=0.0):
        """Time-varying reference: ee_traj/base_traj/base_vel are (N+1, 3), one row per shooting node."""
        th_ref = self.th_nom if th_ref is None else np.asarray(th_ref, dtype=float)
        qv = np.array([0.0, 0.0, np.sin(yaw / 2)])
        for k in range(self.prm.N + 1):
            y = np.r_[ee_traj[k], base_traj[k], qv, base_vel[k], np.zeros(3), th_ref, np.zeros(3)]
            if k < self.prm.N:
                self.solver.set(k, "yref", np.r_[y, u_hover[: self.nf], np.zeros(3) if self.thcmd_rel else th_ref])
            else:
                self.solver.set(k, "yref", y)

    def warm_start(self, x0, u0):
        for k in range(self.prm.N + 1):
            self.solver.set(k, "x", x0)
        for k in range(self.prm.N):
            self.solver.set(k, "u", u0)

    def _push_params(self):
        p = np.r_[self.f_ee_param, self.f_dist_param]
        for k in range(self.prm.N + 1):
            self.solver.set(k, "p", p)

    def set_contact_force(self, f_world):
        """External EE force the MPC should anticipate over the horizon (e.g. measured or commanded contact)."""
        self.f_ee_param = np.asarray(f_world, dtype=float)
        self._push_params()

    def set_disturbance(self, f_world):
        """Estimated disturbance force on the base (world frame), e.g. from DisturbanceObserver."""
        self.f_dist_param = np.asarray(f_world, dtype=float)
        self._push_params()

    def __call__(self, x0):
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        t0 = time.perf_counter()
        status = self.solver.solve()
        self.solve_times.append(time.perf_counter() - t0)
        self.status_counts[status] = self.status_counts.get(status, 0) + 1
        u = self.solver.get(0, "u")
        # shift the solution forward one stage for the next warm start
        N = self.prm.N
        for k in range(N - 1):
            self.solver.set(k, "x", self.solver.get(k + 1, "x"))
            self.solver.set(k, "u", self.solver.get(k + 1, "u"))
        self.solver.set(N - 1, "x", self.solver.get(N, "x"))
        return u, status

    def predicted_ee(self):
        return np.array([np.asarray(self.wb.f_ee(self.solver.get(k, "x"))).ravel() for k in range(self.prm.N + 1)])


# ---------------------------------------------------------------------------------------------
# Glue to the MuJoCo plant
# ---------------------------------------------------------------------------------------------
def state_from_mujoco(model, data, qa_idx, qv_idx):
    """MuJoCo free joint: qpos [p, q wxyz], qvel [v world, w body]. Matches the MPC state."""
    return np.r_[data.qpos[0:7], data.qvel[0:3], data.qvel[3:6], data.qpos[qa_idx], data.qvel[qv_idx]]


class ArmReference:
    """Integrates the MPC's arm reference model at physics rate; the servos track its output."""

    def __init__(self, th0, wn):
        self.th = np.array(th0, dtype=float)
        self.thd = np.zeros(3)
        self.wn = wn

    def step(self, th_cmd, dt):
        thdd = self.wn ** 2 * (np.asarray(th_cmd) - self.th) - 2 * self.wn * self.thd
        self.thd += thdd * dt
        self.th += self.thd * dt
        return self.th, self.thd


class DisturbanceObserver:
    """Estimates the unmodelled force on the base from the mismatch between measured and predicted velocity.

    Every control step: predict v_k+1 with the nominal model (including the current estimate), then
    d <- d + gain * m * (v_meas - v_pred) / dt. A first-order estimator in the spirit of the L1 / momentum
    observers used on real aerial manipulators; it absorbs mass/thrust mismatch, wind and contact-force error.
    """

    def __init__(self, wb: "WholeBodyModel", dt: float, gain: float = 0.08, limit: float = 6.0):
        self.wb, self.dt, self.gain, self.limit = wb, dt, gain, limit
        self.d = np.zeros(3)
        self.prev = None

    def update(self, x_meas, u_applied, f_ee):
        if self.prev is not None:
            x0, u0, fe0 = self.prev
            xdot = np.asarray(self.wb.f_dyn(x0, u0, np.r_[fe0, self.d])).ravel()
            v_pred = x0[7:10] + self.dt * xdot[7:10]
            self.d = np.clip(self.d + self.gain * self.wb.mass * (x_meas[7:10] - v_pred) / self.dt, -self.limit, self.limit)
        self.prev = (np.array(x_meas), np.array(u_applied), np.array(f_ee))
        return self.d

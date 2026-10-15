"""LQR with integral action (LQI) for the multirotor + arm, from the linearized MuJoCo model.

Linearization point: hover, level, arm at a nominal pose. State (MuJoCo tangent space, 22 for a quad):
[dq (9: position, rotation vector, 3 arm joints), qvel (9), rotor thrust states (n)] plus 3 integral
states on the base-position error. Inputs: the n rotor thrusts (the arm is commanded separately,
e.g. by IK, and enters the LQR only as measured state).

What this captures and what it does not:
  * the coupling at the linearization point (arm offset -> CoM moment, arm velocity -> reaction)
  * nothing about thrust limits (commands are just clipped) and nothing about the arm pose
    changing the dynamics -- one gain for one pose; gain scheduling would be the fix
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from .trim import hover_trim, linearize


class LQIController:
    def __init__(self, model, cfg, th_nominal, pos=(0.0, 0.0, 1.0), q_pos=40.0, q_rot=20.0, q_vel=8.0, q_rate=1.0,
                 q_int=20.0, r=1.0, schedule_step=None, arm_feedback=True):
        """schedule_step: relinearize (gain scheduling) when the arm moves this far [rad] from the last
        linearization pose; None = one fixed gain. arm_feedback=False zeroes the gain on arm states."""
        self.model, self.cfg = model, cfg
        self.weights = dict(q_pos=q_pos, q_rot=q_rot, q_vel=q_vel, q_rate=q_rate, q_int=q_int, r=r)
        self.schedule_step, self.arm_feedback = schedule_step, arm_feedback
        self.pos = np.asarray(pos, dtype=float)
        self.z = np.zeros(3)
        self.n_relinearize = 0
        self._design(np.asarray(th_nominal, dtype=float))
        self.f_hat = self.f_trim.copy()                  # internal estimate of the rotor states (from commands)

    def _design(self, th):
        model, cfg = self.model, self.cfg
        w = self.weights
        q_pos, q_rot, q_vel, q_rate, q_int, r = w["q_pos"], w["q_rot"], w["q_vel"], w["q_rate"], w["q_int"], w["r"]
        n = cfg.n_rotors
        self.n, self.dt = n, model.opt.timestep
        self.th_lin = th.copy()
        self.trim = hover_trim(model, cfg, th, pos=self.pos)
        A, B = linearize(model, self.trim)
        B = B[:, :n]                                     # rotor thrust inputs only
        nv, na = model.nv, model.na
        nx = A.shape[0]
        C = np.zeros((3, nx)); C[:, 0:3] = np.eye(3)     # position part of dq
        Aa = np.block([[A, np.zeros((nx, 3))], [self.dt * C, np.eye(3)]])
        Ba = np.vstack([B, np.zeros((3, n))])
        q = np.zeros(nx + 3)
        q[0:3], q[3:6] = q_pos, q_rot                    # position, attitude (arm joints left at 0)
        q[nv:nv + 3], q[nv + 3:nv + 6] = q_vel, q_rate   # linear velocity, body rates
        q[nx:] = q_int
        P = solve_discrete_are(Aa, Ba, np.diag(q), r * np.eye(n))
        self.K = np.linalg.solve(r * np.eye(n) + Ba.T @ P @ Ba, Ba.T @ P @ Aa)
        if not self.arm_feedback:
            self.K[:, 6:9] = 0.0                          # arm joint angles
            self.K[:, nv + 6:nv + 9] = 0.0                # arm joint rates
        self.cl_spectral_radius = float(np.max(np.abs(np.linalg.eigvals(Aa - Ba @ self.K))))
        self.nv, self.na, self.nx = nv, na, nx
        self.qpos_trim = self.trim.qpos.copy()
        self.f_trim = self.trim.ctrl[:n].copy()
        self.tau = cfg.motor_tau
        self._dq = np.zeros(nv)
        self.n_relinearize += 1

    def __call__(self, qpos_meas, qvel_meas, p_ref, v_ref=np.zeros(3), th_meas=None):
        if self.schedule_step is not None and th_meas is not None and np.max(np.abs(th_meas - self.th_lin)) > self.schedule_step:
            self._design(np.asarray(th_meas, dtype=float))   # gain scheduling on the arm pose
        qref = self.qpos_trim.copy()
        qref[0:3] = p_ref
        mujoco.mj_differentiatePos(self.model, self._dq, 1.0, qref, qpos_meas)
        dv = qvel_meas.copy()
        dv[0:3] -= v_ref
        self.z = np.clip(self.z + self.dt * self._dq[0:3], -0.5, 0.5)
        dx = np.r_[self._dq, dv, self.f_hat - self.f_trim, self.z]
        f = np.clip(self.f_trim - self.K @ dx, 0.0, self.cfg.thrust_max)
        self.f_hat += (f - self.f_hat) * self.dt / self.tau
        return f

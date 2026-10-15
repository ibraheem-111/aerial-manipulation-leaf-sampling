"""Baseline flight controller: geometric SE(3) tracking (Lee, Leok, McClamroch, CDC 2010) plus
thrust allocation, with the arm on independent position servos.

This is the "plain" baseline the arm-aware controllers will be compared against. Mass and inertia
come from MuJoCo's joint-space inertia matrix at the current arm pose (whole system about the base
origin). With `com_ff=True` it also cancels the static gravity moment of the arm-shifted CoM, so it
knows where the arm is; it never models how the arm is moving (reaction torques stay a disturbance).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import UAMConfig

E3 = np.array([0.0, 0.0, 1.0])
G = 9.81


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def quat_to_R(q_wxyz):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q_wxyz)
    return R.reshape(3, 3)


@dataclass
class Gains:
    pos_wn: float = 2.5      # position loop natural frequency [rad/s]
    pos_zeta: float = 0.9
    pos_ki: float = 0.6      # integral gain relative to kx [1/s]
    att_wn: float = 18.0     # attitude loop natural frequency [rad/s]
    att_zeta: float = 0.8
    i_limit: float = 0.5     # integral clamp [m*s]


class GeometricController:
    def __init__(self, model, cfg: UAMConfig, gains: Gains | None = None, com_ff: bool = True):
        self.model, self.cfg, self.g = model, cfg, gains or Gains()
        self.com_ff = com_ff
        self.base_id = model.body("base").id
        self.A = cfg.allocation_matrix()
        self.A_pinv = np.linalg.pinv(self.A)
        self.int_ex = np.zeros(3)
        self._M = np.zeros((model.nv, model.nv))

    def reset(self):
        self.int_ex[:] = 0.0

    def mass_props(self, data):
        mujoco.mj_fullM(self.model, self._M, data.qM)
        return self._M[0, 0], self._M[3:6, 3:6]

    def __call__(self, data, p_d, yaw_d=0.0, v_d=np.zeros(3), a_d=np.zeros(3), dt=None):
        dt = dt or self.model.opt.timestep
        m, J = self.mass_props(data)
        p, v = data.qpos[0:3], data.qvel[0:3]
        R = quat_to_R(data.qpos[3:7])
        Om = data.qvel[3:6]

        # Translational loop -> desired force in world frame
        kx = m * self.g.pos_wn ** 2
        kv = 2 * self.g.pos_zeta * self.g.pos_wn * m
        ex, ev = p - p_d, v - v_d
        self.int_ex = np.clip(self.int_ex + ex * dt, -self.g.i_limit, self.g.i_limit)
        F = -kx * ex - kv * ev - kx * self.g.pos_ki * self.int_ex + m * G * E3 + m * a_d

        # Desired attitude from force direction and yaw
        b3d = F / np.linalg.norm(F)
        b1c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        b2d = np.cross(b3d, b1c)
        b2d /= np.linalg.norm(b2d)
        Rd = np.column_stack([np.cross(b2d, b3d), b2d, b3d])
        f = F @ (R @ E3)

        # Rotational loop -> body moment about the base origin
        Jd = np.diag(J)
        kR = Jd * self.g.att_wn ** 2
        kW = Jd * 2 * self.g.att_zeta * self.g.att_wn
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        M = -kR * eR - kW * Om + np.cross(Om, J @ Om)
        if self.com_ff:
            # Cancel the gravity moment of the offset CoM (arm pose dependent) about the base origin.
            r = data.subtree_com[self.base_id] - data.xpos[self.base_id]
            tau_g_world = np.cross(r, np.array([0.0, 0.0, -m * G]))
            M = M - R.T @ tau_g_world

        thrusts = np.clip(self.A_pinv @ np.r_[f, M], 0.0, self.cfg.thrust_max)
        return thrusts, dict(eR=eR, ex=ex, f=f, M=M, Rd=Rd)

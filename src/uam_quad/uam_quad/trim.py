"""Hover trim and linearization of the multirotor + arm model.

`hover_trim` finds rotor thrusts and servo setpoints that hold the whole system still at a given
arm pose: it asks MuJoCo's inverse dynamics for the generalized forces needed at qvel = qacc = 0,
maps the base part onto rotor thrusts through the allocation matrix, and turns the arm part into
servo setpoints. `linearize` then wraps mjd_transitionFD around that point.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import UAMConfig


@dataclass
class Trim:
    qpos: np.ndarray
    ctrl: np.ndarray        # [rotor thrusts..., servo setpoints x3]
    act: np.ndarray         # rotor filter states (= thrusts at trim)
    residual: float         # |unmatched base wrench| after allocation [N or N*m]
    qacc_norm: float        # |qacc| after applying the trim (should be ~0)


def arm_qpos_index(model) -> np.ndarray:
    return np.array([model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)])


def hover_trim(model, cfg: UAMConfig, q_arm, pos=(0.0, 0.0, 1.0)) -> Trim:
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq)
    qpos[0:3] = pos
    qpos[3] = 1.0  # level attitude, quaternion w first
    qpos[arm_qpos_index(model)] = q_arm
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    tau = data.qfrc_inverse.copy()

    # Free joint: dofs 0-2 = world-frame force, 3-5 = body-frame torque about the base origin.
    # Rotors only make body-z force, so at level attitude we match [Fz, Mx, My, Mz].
    A = cfg.allocation_matrix()
    wrench = np.array([tau[2], tau[3], tau[4], tau[5]])
    thrusts, *_ = np.linalg.lstsq(A, wrench, rcond=None)
    residual = float(np.linalg.norm(np.r_[tau[0:2], A @ thrusts - wrench]))

    n = cfg.n_rotors
    servo_set = np.asarray(q_arm, dtype=float) + tau[6:9] / cfg.servo_kp  # kp*(ctrl - q) = tau
    ctrl = np.r_[thrusts, servo_set]
    act = thrusts.copy()

    data.ctrl[:] = ctrl
    data.act[:n] = act
    mujoco.mj_forward(model, data)
    return Trim(qpos, ctrl, act, residual, float(np.linalg.norm(data.qacc)))


def linearize(model, trim: Trim, eps: float = 1e-6):
    """Discrete-time x_{k+1} = A x_k + B u_k about the trim (MuJoCo tangent-space state).

    State ordering: [dq (nv), qvel (nv), act (na)]; dq uses a 3-vector for the base rotation.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = trim.qpos
    data.ctrl[:] = trim.ctrl
    data.act[:len(trim.act)] = trim.act
    mujoco.mj_forward(model, data)
    nx = 2 * model.nv + model.na
    A = np.zeros((nx, nx))
    B = np.zeros((nx, model.nu))
    mujoco.mjd_transitionFD(model, data, eps, True, A, B, None, None)
    return A, B


def controllability_rank(A: np.ndarray, B: np.ndarray, tol: float = 1e-8) -> int:
    n = A.shape[0]
    blocks, Ak_B = [], B
    for _ in range(n):
        blocks.append(Ak_B)
        Ak_B = A @ Ak_B
    return int(np.linalg.matrix_rank(np.hstack(blocks), tol=tol))

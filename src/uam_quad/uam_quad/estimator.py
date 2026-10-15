"""Disturbance-force estimator (Kalman filter) for the whole-body MPC.

What it estimates: one force vector d (world frame, newtons) acting on the drone that the MPC's
model does not contain -- wind, extra or missing mass, thrust loss, contact-force error. The MPC then
plans with d included (WholeBodyMPC.set_disturbance), which gives offset-free behaviour.

Model per world axis (x, y, z are independent), sampled every dt:
    p_k+1 = p_k + dt v_k + dt^2/2 (a_k + d_k/m)
    v_k+1 = v_k + dt (a_k + d_k/m)
    d_k+1 = d_k + w_k              (d is a slowly drifting constant, random walk)
    measurement z_k = [p_k, v_k] + noise
where a_k is the acceleration the nominal model predicts (uam_quad.wbmpc f_dyn) from the measured
state, the *estimated* rotor thrusts (commands passed through the motor-lag model) and the measured
contact force. Everything the model already explains is in a_k, so d only picks up what it misses.
"""
from __future__ import annotations

import numpy as np


class DisturbanceKF:
    def __init__(self, wb, dt=0.02, motor_tau=0.02, sigma_p=0.002, sigma_v=0.02, sigma_acc=1.0, sigma_d=0.3,
                 d_limit=8.0):
        """wb: WholeBodyModel (for the nominal acceleration and the mass).
        sigma_p, sigma_v: measurement noise [m, m/s]; sigma_acc: unmodelled acceleration [m/s^2];
        sigma_d: how fast the disturbance may drift [N/sqrt(s)]."""
        self.wb, self.dt, self.m = wb, dt, wb.mass
        self.n = wb.n
        self.alpha = 1.0 - np.exp(-dt / motor_tau)          # exact first-order motor response over dt
        self.d_limit = d_limit
        m = self.m
        self.F = np.array([[1, dt, 0.5 * dt ** 2 / m], [0, 1, dt / m], [0, 0, 1]])
        self.B = np.array([0.5 * dt ** 2, dt, 0.0])
        self.H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        self.Q = np.diag([(0.5 * sigma_acc * dt ** 2) ** 2, (sigma_acc * dt) ** 2, sigma_d ** 2 * dt])
        self.R = np.diag([sigma_p ** 2, sigma_v ** 2])
        self.s = np.zeros((3, 3))                           # rows: [p, v, d], columns: x, y, z
        self.P = [np.diag([1e-4, 1e-3, 1.0]) for _ in range(3)]
        self.f_hat = None
        self.a_prev = None
        self.started = False

    @property
    def d(self):
        return self.s[2].copy()

    def update(self, x_meas, u_cmd, f_ee_meas):
        """Call once per control step with the measured state (MPC format), the command just sent
        (n thrusts + 3 joint targets), and the measured EE contact force (world). Returns d [N]."""
        x_meas = np.asarray(x_meas, dtype=float)
        p, v = x_meas[0:3], x_meas[7:10]
        if not self.started:
            self.s[0], self.s[1] = p, v
            self.f_hat = np.asarray(u_cmd[: self.n], dtype=float).copy()
            self.started = True
        else:
            for i in range(3):                               # predict with last step's model acceleration
                self.s[:, i] = self.F @ self.s[:, i] + self.B * self.a_prev[i]
                self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q
                z = np.array([p[i], v[i]])
                S = self.H @ self.P[i] @ self.H.T + self.R
                K = self.P[i] @ self.H.T @ np.linalg.inv(S)
                self.s[:, i] += K @ (z - self.H @ self.s[:, i])
                self.P[i] = (np.eye(3) - K @ self.H) @ self.P[i]
            self.s[2] = np.clip(self.s[2], -self.d_limit, self.d_limit)
        # thrust the rotors actually produce during the next interval (motor lag), then model acceleration
        self.f_hat = self.f_hat + self.alpha * (np.asarray(u_cmd[: self.n]) - self.f_hat)
        u_eff = np.r_[self.f_hat, np.asarray(u_cmd[self.n:])]
        xdot = np.asarray(self.wb.f_dyn(x_meas, u_eff, np.r_[np.asarray(f_ee_meas, dtype=float), np.zeros(3)])).ravel()
        self.a_prev = xdot[7:10]
        return self.d

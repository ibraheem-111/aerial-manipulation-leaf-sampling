"""MuJoCo as the physics simulator behind PX4 SITL (PX4's Simulator MAVLink API).

PX4 connects over TCP to port 4560. The protocol is lockstep: PX4's clock only advances when it gets
a HIL_SENSOR message, so it always runs at simulation speed.

Each step (250 Hz, the IMU rate PX4 expects; two 2 ms physics steps):
  1. send HIL_SENSOR with IMU, magnetometer and barometer (+ HIL_GPS at 10 Hz)
  2. wait for HIL_ACTUATOR_CONTROLS: PX4's motor outputs, normalized 0..1
  3. map them to rotor thrust and step MuJoCo

Frames. MuJoCo uses a z-up world (x forward = north, y left, z up) and a FLU body. PX4 uses NED and
FRD. Both conversions flip y and z: v_NED = T v_W and v_FRD = T v_FLU with T = diag(1, -1, -1), so
R_NED_FRD = T R T.

Thrust. PX4 maps desired thrust to a motor command through THR_MDL_FAC (k):
    thrust / f_max = k u^2 + (1 - k) u
The bridge uses the same k, so PX4's inversion is exact. The motor lag stays in MuJoCo's rotor
actuators (20 ms filter).

The arm is not driven by PX4. `arm_cmd` (set over ROS 2 by the MPC, see scripts/px4_sim.py --ros) goes
through the same critically damped reference model the MPC assumes (wbmpc.ArmReference), standing in for
the servo's motion profile; the servos track it with gravity feedforward.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from .model import UAMConfig
from .wbmpc import ArmReference

T_FLIP = np.array([1.0, -1.0, -1.0])
G = 9.80665

# Home location: PX4's default SITL home (Zurich), so its world magnetic model check passes.
HOME_LAT, HOME_LON, HOME_ALT = 47.397742, 8.545594, 488.0
MAG_NED_GAUSS = np.array([0.21523, 0.0, 0.42741])       # Earth field at HOME (same values as jMAVSim)
R_EARTH = 6378137.0

# HIL_SENSOR fields_updated bits (PX4 SimulatorMavlink SensorSource)
ACCEL, GYRO, MAG, BARO = 0b111, 0b111000, 0b111000000, 0b1101000000000


def px4_params(cfg: UAMConfig, thr_mdl_fac: float) -> dict:
    """PX4 parameters that describe this airframe (passed to PX4 as PX4_PARAM_* env variables)."""
    p = {"CA_ROTOR_COUNT": cfg.n_rotors, "THR_MDL_FAC": thr_mdl_fac,
         "COM_RC_IN_MODE": 4,           # stick input disabled: no RC transmitter in simulation
         "NAV_DLL_ACT": 0,              # no ground-station-loss failsafe (no QGroundControl needed)
         "IMU_INTEG_RATE": 250}
    pos, spins = cfg.rotor_positions(), cfg.spins()
    for i in range(cfg.n_rotors):
        p[f"CA_ROTOR{i}_PX"] = round(float(pos[i, 0]), 4)
        p[f"CA_ROTOR{i}_PY"] = round(float(-pos[i, 1]), 4)   # FLU y (left) -> FRD y (right)
        p[f"CA_ROTOR{i}_KM"] = round(float(spins[i] * cfg.yaw_coef), 4)   # + = CCW seen from above
    return p


class MujocoPX4Bridge:
    def __init__(self, model, data, cfg: UAMConfig, arm_target, thr_mdl_fac=1.0, port=4560,
                 sensor_hz=250, gps_hz=10, noise=True, seed=0, arm_wn=25.0):
        from pymavlink import mavutil
        self.m, self.d, self.cfg = model, data, cfg
        self.n = cfg.n_rotors
        self.k = thr_mdl_fac
        self.arm_cmd = np.asarray(arm_target, dtype=float).copy()
        self.arm_ref = ArmReference(self.arm_cmd, arm_wn)
        self.qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
        self.qv = [model.jnt_dofadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
        self.sub = int(round(1.0 / sensor_hz / model.opt.timestep))      # physics steps per sensor message
        self.dt = self.sub * model.opt.timestep
        self.gps_every = int(round(sensor_hz / gps_hz))
        self.noise, self.rng = noise, np.random.default_rng(seed)
        self.u = np.zeros(self.n)
        self.armed = False
        self.got_actuators = False
        self.k_step = 0
        self.t_us = 0
        print(f"[bridge] waiting for PX4 on tcp port {port} ...", flush=True)
        self.conn = mavutil.mavlink_connection(f"tcpin:0.0.0.0:{port}", source_system=1, source_component=51)
        self.gyro_adr = model.sensor_adr[model.sensor("gyro").id]
        self.acc_adr = model.sensor_adr[model.sensor("accel").id]

    # --- sensors ----------------------------------------------------------------------------
    def _R_ned_frd(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[3:7])
        return T_FLIP[:, None] * R.reshape(3, 3) * T_FLIP[None, :]

    def _n(self, s, size=3):
        return self.rng.normal(0.0, s, size) if self.noise else np.zeros(size)

    def send_sensors(self):
        d = self.d
        gyro = T_FLIP * d.sensordata[self.gyro_adr:self.gyro_adr + 3] + self._n(0.003)
        acc = T_FLIP * d.sensordata[self.acc_adr:self.acc_adr + 3] + self._n(0.03)
        mag = self._R_ned_frd().T @ MAG_NED_GAUSS + self._n(0.002)
        alt = HOME_ALT + d.qpos[2]
        press = 1013.25 * (1.0 - 0.0065 * alt / 288.15) ** 5.2561 + float(self._n(0.01, 1)[0])
        temp = 15.0 - 0.0065 * alt
        self.conn.mav.hil_sensor_send(
            self.t_us, *acc, *gyro, *mag, press, 0.0, alt, temp, ACCEL | GYRO | MAG | BARO, 0)
        if self.k_step % self.gps_every == 0:
            p_ned, v_ned = T_FLIP * d.qpos[0:3], T_FLIP * d.qvel[0:3]
            lat = HOME_LAT + np.degrees(p_ned[0] / R_EARTH)
            lon = HOME_LON + np.degrees(p_ned[1] / (R_EARTH * np.cos(np.radians(HOME_LAT))))
            vh = np.hypot(v_ned[0], v_ned[1])
            cog = int(np.degrees(np.arctan2(v_ned[1], v_ned[0])) % 360 * 100) if vh > 0.1 else 65535
            self.conn.mav.hil_gps_send(
                self.t_us, 3, int(lat * 1e7), int(lon * 1e7), int((HOME_ALT - p_ned[2]) * 1000),
                30, 40, int(vh * 100), int(v_ned[0] * 100), int(v_ned[1] * 100), int(v_ned[2] * 100),
                cog, 12, 0)

    # --- actuators --------------------------------------------------------------------------
    def receive_actuators(self):
        """Block (lockstep) for PX4's motor outputs once it has started sending them."""
        while True:
            msg = self.conn.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=self.got_actuators,
                                       timeout=1.0 if self.got_actuators else None)
            if msg is None:
                return
            self.got_actuators = True
            self.armed = bool(msg.mode & 128)            # MAV_MODE_FLAG_SAFETY_ARMED
            self.u = np.clip(np.array(msg.controls[: self.n]), 0.0, 1.0) if self.armed else np.zeros(self.n)
            return

    def joint_state(self):
        return self.d.qpos[self.qa].copy(), self.d.qvel[self.qv].copy()

    def thrusts(self):
        rel = self.k * self.u ** 2 + (1.0 - self.k) * self.u
        return rel * self.cfg.thrust_max

    # --- main step --------------------------------------------------------------------------
    def step(self):
        self.send_sensors()
        self.receive_actuators()
        f = self.thrusts()
        for _ in range(self.sub):
            th_r, _ = self.arm_ref.step(self.arm_cmd, self.m.opt.timestep)
            self.d.ctrl[: self.n] = f
            self.d.ctrl[self.n:] = th_r + self.d.qfrc_bias[self.qv] / self.cfg.servo_kp
            mujoco.mj_step(self.m, self.d)
        self.t_us += int(round(self.dt * 1e6))
        self.k_step += 1


def run(model, data, bridge: MujocoPX4Bridge, realtime=1.0, viewer=False, log_every_s=2.0, on_step=None):
    """Run the bridge loop. realtime=1.0 paces to the wall clock (0 = as fast as possible)."""
    v = None
    if viewer:
        import mujoco.viewer
        v = mujoco.viewer.launch_passive(model, data)
    t0 = time.perf_counter()
    next_log = 0.0
    while v is None or v.is_running():
        bridge.step()
        if on_step is not None:
            on_step(bridge)
        sim_t = bridge.t_us * 1e-6
        if v is not None and bridge.k_step % 8 == 0:
            v.sync()
        if realtime > 0:
            lag = sim_t / realtime - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
        if sim_t >= next_log:
            next_log += log_every_s
            p = data.qpos[0:3]
            print(f"[bridge] t={sim_t:7.2f}s armed={bridge.armed} pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                  f"u={np.round(bridge.u, 2)}", flush=True)

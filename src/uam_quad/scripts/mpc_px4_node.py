"""Whole-body MPC as a ROS 2 node flying PX4 (offboard thrust + body rates) and commanding the arm.

    terminal 1:  python scripts/px4_sim.py --ros             (MuJoCo + PX4 SITL)
    terminal 2:  MicroXRCEAgent udp4 -p 8888                 (PX4 <-> ROS 2)
    terminal 3:  python scripts/mpc_px4_node.py              (this node)

Reads   /fmu/out/vehicle_odometry  (PX4's state estimate: NED position/velocity, FRD->NED attitude, FRD rates)
        /fmu/out/vehicle_status(_v1)
        /arm/joint_states
Writes  /fmu/in/offboard_control_mode   (body_rate = true, streamed as PX4's keep-alive)
        /fmu/in/vehicle_rates_setpoint  (FRD body rates + normalized collective thrust)
        /fmu/in/vehicle_command         (offboard mode, arm, land)
        /arm/joint_command

Frames: the MPC works in a z-up world (x north, y west, z up) and a FLU body; PX4 uses NED and FRD. Both
conversions flip y and z (see uam_quad.px4_bridge).

Task (time from arming): climb 1 m (0-5 s), hold, move the gripper 10 cm forward and 5 cm up (8-10 s), hold,
back (14-16 s), hold, land at --duration. Tracking is measured against MuJoCo ground truth (/sim/ee_position).
"""
import argparse
import json
import os

import numpy as np
import rclpy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleOdometry, VehicleRatesSetpoint, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState

from uam_quad import f450_stock, load_model, with_omx_arm
from uam_quad.paths import BUILD, RUNS
from uam_quad.wbmpc import MPCParams, WholeBodyMPC

T_FLIP = np.array([1.0, -1.0, -1.0])
TH_NOM = np.array([-0.6, 1.0, 0.4])
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def min_jerk(a, b, T, t):
    s = np.clip(t / T, 0.0, 1.0)
    return a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


class MpcPx4Node(Node):
    def __init__(self, a):
        super().__init__("uam_mpc")
        self.a = a
        cfg = with_omx_arm(f450_stock())
        model, self.cfg = load_model(cfg)
        prm = MPCParams(arm_wn=25.0, w_base=200.0, input_mode="rates", tau_rate=a.tau_rate,
                        codegen_dir=os.path.join(BUILD, "uam_wbmpc_px4"))
        self.prm = prm
        self.mpc = WholeBodyMPC(model, self.cfg, prm, th_nominal=TH_NOM)
        self.wb = self.mpc.wb
        self.T_max = self.wb.thrust_max_total
        self.u_hover = np.r_[self.wb.mass * 9.81, 0.0, 0.0, 0.0, TH_NOM]
        # gripper offset from the base origin at the nominal arm pose, level attitude
        x_lvl = np.r_[0, 0, 0, 1, 0, 0, 0, np.zeros(6), TH_NOM, np.zeros(3)]
        self.ee_off = np.asarray(self.wb.f_ee(x_lvl)).ravel()

        self.pub_mode = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self.pub_rates = self.create_publisher(VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        self.pub_arm = self.create_publisher(JointState, "/arm/joint_command", 10)
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.on_odom, PX4_QOS)
        for topic in ("/fmu/out/vehicle_status", "/fmu/out/vehicle_status_v1"):
            self.create_subscription(VehicleStatus, topic, self.on_status, PX4_QOS)
        self.create_subscription(JointState, "/arm/joint_states", self.on_joints, 10)
        self.create_subscription(PointStamped, "/sim/ee_position", self.on_ee_truth, 10)

        self.odom = None
        self.th, self.thd = TH_NOM.copy(), np.zeros(3)
        self.armed = self.offboard = False
        self.k = 0
        self.t_arm = None
        self.p0 = None
        self.ee_truth = None
        self.log = []
        self.done = False
        self.create_timer(0.02, self.tick)                       # 50 Hz
        self.get_logger().info(f"MPC ready (rates mode, tau_rate={a.tau_rate}); waiting for PX4 odometry")

    # --- inputs ---------------------------------------------------------------------------
    def on_odom(self, m):
        self.odom = m

    def on_status(self, m):
        self.armed = m.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.offboard = m.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def on_joints(self, m):
        self.th = np.asarray(m.position[:3], dtype=float)
        self.thd = np.asarray(m.velocity[:3], dtype=float) if len(m.velocity) >= 3 else np.zeros(3)

    def on_ee_truth(self, m):
        self.ee_truth = np.array([m.point.x, m.point.y, m.point.z])

    def state(self):
        o = self.odom
        p = T_FLIP * np.asarray(o.position, dtype=float)
        qn = np.asarray(o.q, dtype=float)                       # w x y z, FRD -> NED
        q = np.array([qn[0], qn[1], -qn[2], -qn[3]])            # FLU -> z-up world
        v = T_FLIP * np.asarray(o.velocity, dtype=float)        # velocity_frame NED
        w = T_FLIP * np.asarray(o.angular_velocity, dtype=float)
        return np.r_[p, q / np.linalg.norm(q), v, w, self.th, self.thd]

    # --- outputs --------------------------------------------------------------------------
    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def command(self, cmd, p1=0.0, p2=0.0):
        m = VehicleCommand(command=cmd, param1=float(p1), param2=float(p2), target_system=1, target_component=1,
                           source_system=1, source_component=1, from_external=True, timestamp=self.now_us())
        self.pub_cmd.publish(m)

    def send(self, T, w_flu, q_cmd):
        t = self.now_us()
        self.pub_mode.publish(OffboardControlMode(body_rate=True, timestamp=t))
        r = T_FLIP * np.asarray(w_flu, dtype=float)
        sp = VehicleRatesSetpoint(roll=float(r[0]), pitch=float(r[1]), yaw=float(r[2]), timestamp=t)
        sp.thrust_body = [0.0, 0.0, -float(np.clip(T / self.T_max, 0.0, 1.0))]
        self.pub_rates.publish(sp)
        js = JointState(name=["arm_j1", "arm_j2", "arm_j3"], position=[float(v) for v in q_cmd])
        self.pub_arm.publish(js)

    # --- reference --------------------------------------------------------------------------
    def reference(self, t):
        b_hover = self.p0 + np.array([0, 0, self.a.height])
        base = min_jerk(self.p0, b_hover, 5.0, t)
        ee_home = b_hover + self.ee_off
        ee_reach = ee_home + np.array([0.10, 0.0, 0.05])
        if t < 8.0:
            ee = base + self.ee_off
        elif t < 14.0:
            ee = min_jerk(ee_home, ee_reach, 2.0, t - 8.0)
        else:
            ee = min_jerk(ee_reach, ee_home, 2.0, t - 14.0)
        return ee, base

    # --- main loop ------------------------------------------------------------------------
    def tick(self):
        self.k += 1
        if self.odom is None or self.done:
            return
        if self.t_arm is None:
            self.send(0.0, np.zeros(3), TH_NOM)                  # stream setpoints before switching mode
            if self.k > 50 and self.k % 25 == 0:
                if not self.offboard:
                    self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)     # PX4 custom mode 6 = offboard
                elif not self.armed:
                    self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
            if self.armed and self.offboard:
                x = self.state()
                self.p0 = x[0:3].copy()
                self.mpc.warm_start(x, self.u_hover)
                self.t_arm = self.get_clock().now()
                self.get_logger().info(f"armed in offboard; start {np.round(self.p0, 3)}")
            return
        t = (self.get_clock().now() - self.t_arm).nanoseconds * 1e-9
        if t > self.a.duration:
            self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.finish()
            return
        x = self.state()
        refs = [self.reference(t + j * self.prm.dt) for j in range(self.prm.N + 1)]
        self.mpc.set_trajectory(np.array([r[0] for r in refs]), np.array([r[1] for r in refs]),
                                np.zeros((self.prm.N + 1, 3)), self.u_hover, th_ref=TH_NOM)
        u, status = self.mpc(x)
        self.send(u[0], u[1:4], u[4:7])
        ee_ref, base_ref = refs[0]
        self.log.append(dict(t=t, status=int(status), x=x.tolist(), u=u.tolist(), ee_ref=ee_ref.tolist(),
                             base_ref=base_ref.tolist(),
                             ee_truth=None if self.ee_truth is None else self.ee_truth.tolist()))

    def finish(self):
        self.done = True
        out = os.path.join(RUNS, "px4")
        os.makedirs(out, exist_ok=True)
        L = self.log
        t = np.array([r["t"] for r in L])
        ok = np.mean([r["status"] == 0 for r in L]) * 100
        res = dict(tau_rate=self.a.tau_rate, solver_ok_pct=round(float(ok), 1),
                   solve_ms_median=round(float(np.median(self.mpc.solve_times) * 1e3), 2))
        est = np.array([r["x"][0:3] for r in L]); bref = np.array([r["base_ref"] for r in L])
        sel = t > 6.0
        res["base_rms_mm_estimate"] = round(float(np.sqrt(np.mean(np.sum((est - bref)[sel] ** 2, 1))) * 1e3), 1)
        G = [(r["t"], r["ee_ref"], r["ee_truth"]) for r in L if r["ee_truth"] is not None]
        if G:
            # MuJoCo world -> PX4 local: the bridge's world origin is the drone's start position projected to
            # the ground, PX4's local origin is where the EKF started; compare after removing the start offset
            tt = np.array([g[0] for g in G]); er = np.array([g[1] for g in G]); et = np.array([g[2] for g in G])
            off = et[0] - er[0]
            e = np.linalg.norm(et - off - er, axis=1)
            res["ee_rms_mm_truth_after_6s"] = round(float(np.sqrt(np.mean(e[tt > 6.0] ** 2)) * 1e3), 1)
            res["ee_max_mm_truth_after_6s"] = round(float(e[tt > 6.0].max() * 1e3), 1)
        json.dump(dict(result=res, log=L), open(os.path.join(out, "mpc_px4_run.json"), "w"))
        self.get_logger().info(f"done, landing. {res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-rate", type=float, default=0.05, help="PX4 rate-loop time constant in the MPC model [s]")
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=20.0)
    a = ap.parse_args()
    rclpy.init()
    node = MpcPx4Node(a)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

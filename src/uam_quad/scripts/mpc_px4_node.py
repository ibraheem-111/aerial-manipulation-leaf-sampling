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

Tasks (--task, time from arming):
  reach   climb 1 m (0-5 s), hold, move the end effector 10 cm forward and 5 cm up (8-10 s), back (14-16 s),
          land at --duration. Tracking is measured against MuJoCo ground truth (/sim/ee_position).
  excite  rate-loop identification: hover at 1 m, then from 8 s add seeded random rate steps to the MPC's rate
          command, 6 s per axis (roll, pitch, yaw), land at 28 s. Logs the commanded rate and the true body
          rate (/sim/base_twist, 250 Hz) to runs/uam_quad/px4/rate_id_seed<N>.json.
  push    touch and push on the board of scripts/push_task.py (needs px4_sim.py --scene push): climb (0-6 s),
          then push_task's timeline shifted by +4 s (approach, 2 N, 4 N, release), land at 28 s. Admittance on
          the end-effector force sensor (/sim/ee_force), contact hysteresis, force feedforward into the MPC.
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
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped

T_FLIP = np.array([1.0, -1.0, -1.0])
TH_NOM = np.array([-0.6, 1.0, 0.4])
# PX4 closed rate loop in the MPC's form w_dot = (w_cmd - w) / tau: roll, pitch, yaw [s].
# Roll and pitch: identified by scripts/fit_rate_loop.py from the excite flights (seeds 0, 1; held-out seed 2:
# VAF 95.6 %, 89.7 %). Yaw stays at the 0.05 s placeholder: the identified yaw value (0.446 s) fits poorly
# (held-out VAF 74.7 %; PX4's yaw loop has ~0.1 s delay and gain ~1.35, which this form cannot express), and
# with it the MPC saturated the yaw-rate command in contact and 2 of 4 realistic push runs crashed.
TAU_RATE_ID = (0.034, 0.066, 0.05)
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
        self.create_subscription(TwistStamped, "/sim/base_twist", self.on_twist, 50)
        self.create_subscription(WrenchStamped, "/sim/ee_force", self.on_force, 50)
        self.create_subscription(WrenchStamped, "/sim/ee_force_true", self.on_force_true, 50)
        self.create_subscription(PoseStamped, "/sim/base_pose", self.on_pose_truth, 10)
        self.twist_log, self.odom_rate_log, self.cmd_log, self.force_log, self.tilt_log = [], [], [], [], []
        self.exc = self.make_excitation(a.seed) if a.task == "excite" else []
        if a.task == "push":
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import push_task as pt
            self.pt = pt
            self.F_filt, self.delta, self.kappa, self.below_since = 0.0, 0.0, False, None

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

    def t_now(self):
        return None if self.t_arm is None else (self.get_clock().now() - self.t_arm).nanoseconds * 1e-9

    def on_twist(self, m):
        t = self.t_now()
        if t is not None and self.a.task == "excite":
            a = m.twist.angular
            self.twist_log.append((t, a.x, a.y, a.z))

    def on_force(self, m):
        """End-effector force sensor (250 Hz): normal force along n_hat, low-pass 30 ms."""
        if self.a.task != "push":
            return
        f = np.array([m.wrench.force.x, m.wrench.force.y, m.wrench.force.z])
        F_s = float(-self.pt.N_HAT @ f)
        self.F_filt += (F_s - self.F_filt) * (1 / 250) / 0.03

    def on_force_true(self, m):
        t = self.t_now()
        if t is not None and self.a.task == "push":
            self.force_log.append((t, float(-self.pt.N_HAT @ np.array([m.wrench.force.x, m.wrench.force.y, m.wrench.force.z]))))

    def on_pose_truth(self, m):
        t = self.t_now()
        if t is not None:
            o = m.pose.orientation
            self.tilt_log.append((t, float(np.degrees(np.arccos(np.clip(1 - 2 * (o.x ** 2 + o.y ** 2), -1, 1))))))

    def make_excitation(self, seed):
        """Seeded random steps on one rate axis at a time: 6 s per axis from 8 s, step lengths 0.1-0.35 s."""
        rng = np.random.default_rng(seed)
        amp = (0.6, 0.6, 0.4)                                  # rad/s (roll, pitch, yaw)
        segs, t0 = [], 8.0
        for axis in range(3):
            t = t0
            while t < t0 + 6.0:
                dur = rng.uniform(0.1, 0.35)
                segs.append((t, min(t + dur, t0 + 6.0), axis, amp[axis] * rng.choice([-1.0, 1.0])))
                t += dur
            segs.append((t0 + 6.0, t0 + 6.0, axis, 0.0))
            t0 += 6.0
        return segs

    def excitation(self, t):
        e = np.zeros(3)
        for t0, t1, axis, val in self.exc:
            if t0 <= t < t1:
                e[axis] = val
        return e

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
        tt = self.t_now()
        if tt is not None and self.a.task == "excite":
            self.cmd_log.append((tt, *map(float, w_flu)))
        t = self.now_us()
        self.pub_mode.publish(OffboardControlMode(body_rate=True, timestamp=t))
        r = T_FLIP * np.asarray(w_flu, dtype=float)
        sp = VehicleRatesSetpoint(roll=float(r[0]), pitch=float(r[1]), yaw=float(r[2]), timestamp=t)
        sp.thrust_body = [0.0, 0.0, -float(np.clip(T / self.T_max, 0.0, 1.0))]
        self.pub_rates.publish(sp)
        js = JointState(name=["arm_j1", "arm_j2", "arm_j3"], position=[float(v) for v in q_cmd])
        self.pub_arm.publish(js)

    # --- reference --------------------------------------------------------------------------
    def push_geometry(self):
        """Board and standoff points in PX4's local frame (z up). The bridge starts the hub at (0, 0, 0.2) m in
        the MuJoCo world; PX4's local origin is where the EKF started, p0 at arming."""
        pt = self.pt
        world_start = np.array([0.0, 0.0, 0.2])
        touch = np.array([pt.BOARD_X - pt.TIP_R, 0.0, pt.BOARD_Z]) - world_start + self.p0
        pre = touch - 0.01 * pt.N_HAT
        park = touch - 0.06 * pt.N_HAT - self.ee_off
        hover = np.array([self.p0[0], self.p0[1], park[2]])
        return touch, pre, park, hover

    def reference_push(self, t):
        touch, pre, park, hover = self.push_geometry()
        n = self.pt.N_HAT
        if t < 6.0:                                            # climb
            b = min_jerk(self.p0, hover, 6.0, t)
            return b + self.ee_off, b
        tp = t - 4.0                                           # push_task.py timeline
        if tp < 7.0:
            b = min_jerk(hover, park, 5.0, tp - 2.0)
            return b + self.ee_off, b
        if tp < 9.0:
            return min_jerk(park + self.ee_off, pre, 2.0, tp - 7.0), park
        if tp < 20.0:
            return pre + n * (0.01 + self.delta), park
        return min_jerk(pre + n * (0.01 + self.delta), pre - n * 0.04, 1.5, tp - 20.0), park

    def push_force_loop(self, t):
        """Admittance (eq. 14), contact hysteresis (15) and force feedforward (16), at 50 Hz."""
        pt = self.pt
        tp = t - 4.0
        F_des = pt.force_setpoint(tp)
        if 9.0 <= tp < 20.0:
            self.delta = float(np.clip(self.delta + 0.006 * (F_des - self.F_filt) * 0.02, -0.01, 0.05))
        if not self.kappa and self.F_filt > pt.CONTACT_ON:
            self.kappa, self.below_since = True, None
        elif self.kappa and self.F_filt < 0.1:
            self.below_since = t if self.below_since is None else self.below_since
            if t - self.below_since > 0.1:
                self.kappa = False
        elif self.kappa:
            self.below_since = None
        self.mpc.set_contact_force(-pt.N_HAT * F_des if self.kappa else np.zeros(3))
        return F_des

    def reference(self, t):
        if self.a.task == "push":
            return self.reference_push(t)
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
        F_des = self.push_force_loop(t) if self.a.task == "push" else 0.0
        refs = [self.reference(t + j * self.prm.dt) for j in range(self.prm.N + 1)]
        self.mpc.set_trajectory(np.array([r[0] for r in refs]), np.array([r[1] for r in refs]),
                                np.zeros((self.prm.N + 1, 3)), self.u_hover, th_ref=TH_NOM)
        u, status = self.mpc(x)
        w_cmd = u[1:4] + (self.excitation(t) if self.a.task == "excite" else 0.0)
        self.send(u[0], w_cmd, u[4:7])
        ee_ref, base_ref = refs[0]
        self.log.append(dict(t=t, status=int(status), x=x.tolist(), u=u.tolist(), ee_ref=ee_ref.tolist(),
                             F_des=F_des, F_filt=getattr(self, "F_filt", 0.0), kappa=bool(getattr(self, "kappa", False)),
                             delta=getattr(self, "delta", 0.0),
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
        if self.a.task == "excite":
            json.dump(dict(seed=self.a.seed, tau_rate_model=self.a.tau_rate, cmd=self.cmd_log, truth=self.twist_log,
                           odom_rate=[(r["t"], *r["x"][10:13]) for r in L]),
                      open(os.path.join(out, f"rate_id_seed{self.a.seed}.json"), "w"))
        if self.a.task == "push":
            Fl = np.array(self.force_log); Tl = np.array(self.tilt_log)
            tp, F = Fl[:, 0] - 4.0, Fl[:, 1]
            w2, w4, push = (tp >= 11) & (tp < 15), (tp >= 17) & (tp < 20), (tp >= 11) & (tp < 20)
            res.update(force_mean_2N=round(float(F[w2].mean()), 2),
                       force_rms_err_2N=round(float(np.sqrt(np.mean((F[w2] - 2.0) ** 2))), 2),
                       force_mean_4N=round(float(F[w4].mean()), 2),
                       force_rms_err_4N=round(float(np.sqrt(np.mean((F[w4] - 4.0) ** 2))), 2),
                       contact_lost_pct=round(float(np.mean(F[push] < 0.05) * 100), 1),
                       max_tilt_deg=round(float(Tl[Tl[:, 0] > 1.0, 1].max()), 1))
            json.dump(dict(result=res, log=L, force=self.force_log, tilt=self.tilt_log),
                      open(os.path.join(out, f"push_px4_{self.a.tag}.json"), "w"))
        json.dump(dict(result=res, log=L), open(os.path.join(out, f"mpc_px4_{self.a.task}.json"), "w"))
        self.get_logger().info(f"done, landing. {res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-rate", type=float, nargs="+", default=list(TAU_RATE_ID),
                    help="PX4 rate-loop time constant(s) in the MPC model [s]: one value for all axes or "
                         "roll pitch yaw (default: TAU_RATE_ID)")
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=None, help="land at this time [s] (default: 20 reach, 28 excite/push)")
    ap.add_argument("--task", choices=["reach", "excite", "push"], default="reach")
    ap.add_argument("--seed", type=int, default=0, help="excitation seed (excite)")
    ap.add_argument("--tag", default="run", help="label for the push result file")
    a = ap.parse_args()
    if len(a.tau_rate) not in (1, 3):
        ap.error("--tau-rate takes 1 or 3 values")
    a.tau_rate = a.tau_rate[0] if len(a.tau_rate) == 1 else tuple(a.tau_rate)
    if a.duration is None:
        a.duration = 20.0 if a.task == "reach" else 28.0
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

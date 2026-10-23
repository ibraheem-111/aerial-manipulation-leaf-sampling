"""PX4 SITL flying the MuJoCo F450 + arm (MuJoCo is the physics, PX4 is the flight controller).

Starts the MuJoCo bridge (TCP 4560) and PX4 SITL (v1.16, airframe none_iris with this F450's rotor
geometry passed as PX4_PARAM_* overrides). PX4's MAVLink and uXRCE-DDS (ROS 2) interfaces come up as
in any SITL run:
    MAVLink offboard API   udp 14540        (pymavlink, MAVSDK)
    QGroundControl         udp 14550
    ROS 2 (uXRCE-DDS)      udp 8888         (run: MicroXRCEAgent udp4 -p 8888)

    python scripts/px4_sim.py                 # headless, real time
    python scripts/px4_sim.py --viewer        # with the MuJoCo viewer
    python scripts/px4_sim.py --no-px4        # bridge only; start PX4 yourself
    python scripts/px4_sim.py --ros           # + ROS 2 arm interface and ground truth (needs .venv-ros)

ROS 2 topics with --ros (world frame = MuJoCo: x forward/north, y left, z up):
    /arm/joint_states    sensor_msgs/JointState   arm_j1..3 position, velocity        (published, 100 Hz)
    /arm/joint_command   sensor_msgs/JointState   arm_j1..3 target position           (subscribed)
    /sim/ee_position     geometry_msgs/PointStamped  gripper ground truth             (published, 100 Hz)
    /sim/base_pose       geometry_msgs/PoseStamped   base ground truth                (published, 125 Hz)
    /sim/base_twist      geometry_msgs/TwistStamped  true body rates (angular, body FLU) and world
                                                     velocity (linear)                (published, 250 Hz)
    /sim/ee_force        geometry_msgs/WrenchStamped end-effector contact force from the board, world
                                                     frame, 0.1 N noise with --realistic (--scene push; 250 Hz)

    python scripts/px4_sim.py --ros --scene push    # + the vertical board of scripts/push_task.py
"""
import argparse
import os
import signal
import subprocess
import sys

import mujoco
import numpy as np

from uam_quad import f450_stock, load_model, with_omx_arm
from uam_quad.paths import REPO, RUNS
from uam_quad.px4_bridge import MujocoPX4Bridge, px4_params, run

PX4_DIR = os.path.join(REPO, "external", "PX4-Autopilot")
TH_NOM = (-0.6, 1.0, 0.4)


def start_px4(params, log_path):
    rootfs = os.path.join(PX4_DIR, "build", "px4_sitl_default", "rootfs")
    exe = os.path.join(PX4_DIR, "build", "px4_sitl_default", "bin", "px4")
    if not os.path.exists(exe):
        sys.exit(f"PX4 not built: run  bash setup/build_px4.sh  ({exe} missing)")
    env = dict(os.environ, PX4_SIM_MODEL="none_iris")
    env.update({f"PX4_PARAM_{k}": str(v) for k, v in params.items()})
    log = open(log_path, "w")
    proc = subprocess.Popen([exe, "-d"], cwd=rootfs, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    print(f"[px4] started (pid {proc.pid}), log: {log_path}", flush=True)
    return proc


class RosInterface:
    """Arm joint interface + ground truth over ROS 2, spun in a background thread."""

    def __init__(self, bridge, model, data, force_noise=0.1, seed=0):
        import threading

        import rclpy
        from geometry_msgs.msg import PointStamped, PoseStamped
        from sensor_msgs.msg import JointState
        rclpy.init()
        self.rclpy, self.JointState, self.PointStamped, self.PoseStamped = rclpy, JointState, PointStamped, PoseStamped
        self.node = rclpy.create_node("mujoco_px4_bridge")
        self.bridge, self.m, self.d = bridge, model, data
        self.ee = model.site("ee").id
        self.pub_js = self.node.create_publisher(JointState, "/arm/joint_states", 10)
        self.pub_ee = self.node.create_publisher(PointStamped, "/sim/ee_position", 10)
        self.pub_base = self.node.create_publisher(PoseStamped, "/sim/base_pose", 10)
        from geometry_msgs.msg import TwistStamped, WrenchStamped
        self.TwistStamped, self.WrenchStamped = TwistStamped, WrenchStamped
        self.pub_twist = self.node.create_publisher(TwistStamped, "/sim/base_twist", 10)
        self.pub_force = self.node.create_publisher(WrenchStamped, "/sim/ee_force", 10)
        self.pub_force_true = self.node.create_publisher(WrenchStamped, "/sim/ee_force_true", 10)
        self.board_gid = model.geom("board").id if "board" in [model.geom(i).name for i in range(model.ngeom)] else None
        self.rng, self.force_noise = np.random.default_rng([seed, 11]), force_noise
        self.node.create_subscription(JointState, "/arm/joint_command", self.on_cmd, 10)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def on_cmd(self, msg):
        if len(msg.position) >= 3:
            self.bridge.arm_cmd = np.asarray(msg.position[:3], dtype=float)

    def on_step(self, bridge):
        stamp = self.node.get_clock().now().to_msg()
        tw = self.TwistStamped()                   # every step: 250 Hz
        tw.header.stamp, tw.header.frame_id = stamp, "base"
        tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z = map(float, self.d.qvel[3:6])
        tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = map(float, self.d.qvel[0:3])
        self.pub_twist.publish(tw)
        if self.board_gid is not None:             # end-effector force sensor (normal force from the board)
            from push_task import N_HAT, contact_force
            F_true = contact_force(self.m, self.d, self.board_gid)
            for pub, F in ((self.pub_force, F_true + float(self.rng.normal(0.0, self.force_noise))), (self.pub_force_true, F_true)):
                ws = self.WrenchStamped()
                ws.header.stamp, ws.header.frame_id = stamp, "world"
                ws.wrench.force.x, ws.wrench.force.y, ws.wrench.force.z = map(float, -N_HAT * F)
                pub.publish(ws)
        if bridge.k_step % 2:                      # the rest at 125 Hz
            return
        q, qd = bridge.joint_state()
        js = self.JointState(name=["arm_j1", "arm_j2", "arm_j3"], position=q.tolist(), velocity=qd.tolist())
        js.header.stamp = stamp
        self.pub_js.publish(js)
        pt = self.PointStamped()
        pt.header.stamp, pt.header.frame_id = stamp, "world"
        pt.point.x, pt.point.y, pt.point.z = map(float, self.d.site_xpos[self.ee])
        self.pub_ee.publish(pt)
        ps = self.PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, "world"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, self.d.qpos[0:3])
        w, x, y, z = map(float, self.d.qpos[3:7])
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = w, x, y, z
        self.pub_base.publish(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--no-px4", action="store_true", help="only run the bridge")
    ap.add_argument("--realtime", type=float, default=1.0, help="sim speed vs wall clock; 0 = as fast as possible")
    ap.add_argument("--thr-mdl-fac", type=float, default=1.0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--ros", action="store_true", help="ROS 2 arm interface + ground truth topics")
    ap.add_argument("--scene", choices=["empty", "push"], default="empty", help="push = add the board of push_task.py")
    ap.add_argument("--realistic", action="store_true",
                    help="plant mismatch (+10%% airframe mass, 95%% thrust, 30 ms motors) + gusty wind, as in compare_controllers.py")
    ap.add_argument("--seed", type=int, default=0, help="wind / sensor-noise seed")
    a = ap.parse_args()

    nominal = with_omx_arm(f450_stock())
    cfg = with_omx_arm(f450_stock(base_mass=nominal.base_mass * 1.10, motor_tau=0.03)) if a.realistic else nominal
    cfg.start_height = 0.2                       # resting on the landing legs
    extra = ""
    if a.scene == "push":
        from push_task import SCENE
        extra = SCENE
    model, cfg = load_model(cfg, extra_world=extra)
    if a.realistic:
        model.actuator_gainprm[:cfg.n_rotors, 0] *= 0.95     # rotors give 95 % of commanded thrust
    data = mujoco.MjData(model)
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    data.qpos[qa] = TH_NOM
    mujoco.mj_forward(model, data)

    params = px4_params(nominal if a.realistic else cfg, a.thr_mdl_fac)    # PX4 is told the nominal airframe
    os.makedirs(RUNS, exist_ok=True)
    px4 = None
    if not a.no_px4:
        # PX4 connects to the bridge's TCP server and retries until the bridge is listening
        px4 = start_px4(params, os.path.join(RUNS, "px4_sitl.log"))
    bridge = MujocoPX4Bridge(model, data, cfg, TH_NOM, thr_mdl_fac=a.thr_mdl_fac, noise=not a.no_noise, seed=a.seed)
    # force-sensor noise only in realistic mode, as push_task.py's "fnoise" factor
    ros = RosInterface(bridge, model, data, force_noise=0.1 if a.realistic else 0.0, seed=a.seed) if a.ros else None
    wind_rng, wind = np.random.default_rng([a.seed, 7]), np.zeros(3)
    WIND_MEAN, WIND_DIR = 0.335, np.array([0.3, 0.15, 0.0]) / np.linalg.norm([0.3, 0.15, 0.0])
    base_id = model.body("base").id

    def on_step(b):
        nonlocal wind
        if a.realistic:                        # Ornstein-Uhlenbeck gusts, as in compare_controllers.py
            dt = b.dt
            wind += (-(wind - WIND_MEAN * WIND_DIR) / 1.5) * dt + 1.5 * WIND_MEAN * np.sqrt(2 * dt / 1.5) \
                * wind_rng.normal(size=3) * np.array([1, 1, 0.3])
            data.xfrc_applied[base_id, :3] = wind
        if ros:
            ros.on_step(b)

    try:
        run(model, data, bridge, realtime=a.realtime, viewer=a.viewer, on_step=on_step)
    except KeyboardInterrupt:
        pass
    finally:
        if px4 is not None and px4.poll() is None:
            os.killpg(px4.pid, signal.SIGTERM)
            px4.wait(timeout=5)
            print("[px4] stopped", flush=True)


if __name__ == "__main__":
    main()

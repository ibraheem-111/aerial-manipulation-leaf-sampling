"""MJCF generator for a multirotor (quad or upright hexa) with a 3-DoF pitch arm mounted on top.

Everything physical lives in `UAMConfig`, so switching quad <-> hexa or changing the arm is a
config change. Rotors are real actuators: each produces thrust along the body z axis at its own
site, with a reaction yaw torque and a first-order motor lag. Gravity is on.

Frames: world z up. Base body x forward, y left, z up. Arm joints rotate about body +y, so a
positive joint angle pitches the link tip downward (right-hand rule: +x toward -z).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class UAMConfig:
    # Defaults: DJI F450-class quad (the lab's frame). 450 mm motor-to-motor diagonal; typical
    # build = 2212 ~920KV motors, 10x4.5" props, 3S. Masses and thrust are typical values, not
    # measured on the lab's airframe -- replace them once it is weighed / thrust-stand tested.
    n_rotors: int = 4                      # 4 = quad X, 6 = upright hexa X
    arm_radius: float = 0.225              # rotor distance from body center [m] (F450: 450 mm / 2)
    base_mass: float = 1.1                 # frame + battery + autopilot + companion computer [kg]
    motor_mass: float = 0.06               # per motor + prop [kg]
    base_size: tuple = (0.08, 0.08, 0.035) # half-extents of the central box [m]
    thrust_max: float = 8.0                # per-rotor max thrust [N] (2212 + 1045 prop on 3S, typical)
    yaw_coef: float = 0.016                # rotor drag torque / thrust [N*m/N]
    motor_tau: float = 0.03                # motor first-order time constant [s]
    prop_radius: float = 0.127             # 10 inch props (visual + later airflow model)

    # Arm: 3 pitch joints in the body x-z plane, mounted on top of the base.
    arm_mount: tuple = (0.0, 0.0, 0.05)    # joint-1 position in the base frame [m]
    link_lengths: tuple = (0.20, 0.18, 0.10)
    link_masses: tuple = (0.15, 0.12, 0.06)
    ee_mass: float = 0.05                  # gripper / tool at the tip [kg]
    joint_ranges: tuple = ((-2.6, 1.2), (-2.6, 2.6), (-2.6, 2.6))
    servo_kp: float = 20.0                 # position-servo stiffness [N*m/rad]
    servo_kv: float = 0.6                  # position-servo damping [N*m*s/rad]
    servo_tau_max: float = 6.0             # servo torque limit [N*m] (XM430/540 class)

    timestep: float = 0.002                # 500 Hz physics
    start_height: float = 1.0
    spin_dirs: tuple = field(default=None)  # +1 = CCW seen from above; default alternates

    # Optional overrides (used by the measured-airframe presets such as f450_mrs()).
    rotor_xy: tuple = None                 # explicit rotor (x, y) list, e.g. PX4 motor order
    rotor_z: float = 0.02                  # rotor plane height in the base frame [m]
    base_inertia: tuple = None             # (ixx, iyy, izz) of the airframe about O; if set, base_mass
                                           # is the whole airframe and geom masses are ignored
    mesh_dir: str = None                   # folder with f450_*.obj visual meshes (None = primitives)
    arm_spec: str = None                   # assets/arm/arm_spec.json -> OpenMANIPULATOR-X arm (None = capsules)

    def rotor_positions(self) -> np.ndarray:
        """(n, 3) rotor centers in the base frame, X configuration (no rotor on the x axis)."""
        n = self.n_rotors
        if self.rotor_xy is not None:
            xy = np.asarray(self.rotor_xy, dtype=float)
            return np.column_stack([xy, np.full(n, self.rotor_z)])
        angles = np.pi / n + 2 * np.pi * np.arange(n) / n
        return np.stack([self.arm_radius * np.cos(angles),
                         self.arm_radius * np.sin(angles),
                         np.full(n, self.rotor_z)], axis=1)

    def spins(self) -> np.ndarray:
        if self.spin_dirs is not None:
            return np.asarray(self.spin_dirs, dtype=float)
        return np.array([(-1.0) ** i for i in range(self.n_rotors)])  # alternate CCW / CW

    def allocation_matrix(self) -> np.ndarray:
        """A (4, n): [collective thrust, Mx, My, Mz] about the base origin = A @ rotor_thrusts.

        Thrust f_i acts along body +z at (x_i, y_i): moment = r x F = (y_i f_i, -x_i f_i, 0).
        A CCW-spinning prop (spin +1) pushes the body clockwise: Mz = -spin * yaw_coef * f.
        """
        p = self.rotor_positions()
        return np.vstack([np.ones(self.n_rotors), p[:, 1], -p[:, 0],
                          -self.spins() * self.yaw_coef])


# x500-class frame (PX4 SITL default airframe), kept for comparison.
X500 = dict(arm_radius=0.246, base_mass=1.6, base_size=(0.10, 0.10, 0.04), thrust_max=10.0)

DEFAULT_F450_MESHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "f450", "meshes")


def with_omx_arm(cfg: UAMConfig) -> UAMConfig:
    """Replace the capsule arm with the ROBOTIS OpenMANIPULATOR-X planar 3-DoF arm."""
    from . import omx_arm
    return omx_arm.apply_to_config(cfg)


def f450_stock(**overrides) -> UAMConfig:
    """Generic stock DJI F450 with the DJI E310 propulsion system and a 3S battery.

    Sourced: frame 282 g, 450 mm diagonal, 800-1600 g takeoff weight (DJI F450 spec); E310 2312
    960KV motors 60 g each, 9450 props, max thrust 800 g/rotor @ 12 V (7.85 N), recommended
    350-400 g/rotor (DJI E310 spec). Estimated (not from a datasheet): 4 ESCs 25 g each, props 13 g
    each, 3S 4000 mAh LiPo 330 g, autopilot + GPS + receiver 100 g -> 1.10 kg airframe, so
    1.48 kg with the 3-DoF arm (370 g/rotor, inside DJI's recommended band). Inertia: motors, props
    and ESCs as point masses at the arm tips plus a 0.15 x 0.15 x 0.06 m central box (estimate).
    Motor lag 20 ms and yaw ratio 0.016 m are typical values, not E310 measurements.
    Uses the MRS F450 meshes for visuals and the PX4 quad-X motor order.
    """
    L = 0.225
    a = L * np.sin(np.radians(45))
    m_tip = 4 * (0.060 + 0.013 + 0.025)            # motors + props + ESCs at the arm tips
    m_core = 0.282 + 0.330 + 0.100                  # frame + battery + avionics, treated as a central box
    ixx = 4 * (m_tip / 4) * a ** 2 + m_core * (0.15 ** 2 + 0.06 ** 2) / 12
    izz = 4 * (m_tip / 4) * L ** 2 + m_core * (0.15 ** 2 + 0.15 ** 2) / 12
    cfg = dict(n_rotors=4, arm_radius=L, rotor_xy=((a, -a), (-a, a), (a, a), (-a, -a)),
               spin_dirs=(1, 1, -1, -1), rotor_z=0.0115, base_mass=m_tip + m_core,
               base_inertia=(ixx, ixx, izz), base_size=(0.09, 0.06, 0.025), thrust_max=0.8 * 9.81,
               yaw_coef=0.016, motor_tau=0.02, prop_radius=0.1143, arm_mount=(0.0, 0.0, 0.03),
               mesh_dir=DEFAULT_F450_MESHES)
    cfg.update(overrides)
    return UAMConfig(**cfg)


def f450_mrs(**overrides) -> UAMConfig:
    """DJI F450 with the parameters and meshes of the CTU MRS group's Gazebo model.

    Source: github.com/ctu-mrs/mrs_uav_gazebo_simulator, models/mrs_robots_description/sdf/drones/
    f450.sdf.jinja (BSD-3). Their airframe: 1.64 kg body + 4 x 16 g props, arms 0.23 m, motor constant
    1.05e-5 N/(rad/s)^2 at max 1100 rad/s (12.7 N per rotor, hover at ~56% throttle), moment
    constant 0.06, motor time constants 12.5 ms up / 25 ms down (one 19 ms lag here). Body inertia
    is their 0.25 m x 0.05 m cylinder approximation plus the props as point masses. PX4 quad-X
    motor order: 0 front-right CCW, 1 rear-left CCW, 2 front-left CW, 3 rear-right CW.
    Their build carries an Intel NUC and sensors, so the lab F450 will differ -- weigh it.
    """
    m_body, m_prop, L = 1.64, 0.016, 0.23
    a = L * np.sin(np.radians(45))
    ixx = m_body * (3 * 0.25 ** 2 + 0.05 ** 2) / 12 + 4 * m_prop * a ** 2  # 4 props, each m a^2 about x
    izz = m_body * 0.25 ** 2 / 2 + 4 * m_prop * L ** 2
    cfg = dict(n_rotors=4, arm_radius=L, rotor_xy=((a, -a), (-a, a), (a, a), (-a, -a)),
               spin_dirs=(1, 1, -1, -1), rotor_z=0.0115, base_mass=m_body + 4 * m_prop,
               base_inertia=(ixx, ixx, izz), base_size=(0.09, 0.06, 0.025), thrust_max=1.05e-5 * 1100 ** 2,
               yaw_coef=0.06, motor_tau=0.019, prop_radius=0.127, arm_mount=(0.0, 0.0, 0.03),
               mesh_dir=DEFAULT_F450_MESHES)
    cfg.update(overrides)
    return UAMConfig(**cfg)


def build_mjcf(cfg: UAMConfig, extra_world: str = "") -> str:
    rotors = cfg.rotor_positions()
    spins = cfg.spins()
    bx, by, bz = cfg.base_size
    L = cfg.link_lengths

    rotor_xml = []
    meshes = cfg.mesh_dir is not None
    for i, ((x, y, z), s) in enumerate(zip(rotors, spins)):
        if meshes:
            rotor_xml.append(f"""
      <geom name="motor{i}" type="mesh" mesh="f450_rotor" pos="{x:.9f} {y:.9f} -0.012" mass="0" contype="0" conaffinity="0" rgba="0.12 0.12 0.12 1"/>
      <geom name="prop{i}" type="mesh" mesh="f450_prop" pos="{x:.9f} {y:.9f} {z:.9f}" euler="0 0 {0.6 * i:.2f}" mass="0" contype="0" conaffinity="0" rgba="{'0.9 0.45 0.2 0.85' if s > 0 else '0.25 0.5 0.9 0.85'}"/>
      <site name="rotor{i}" pos="{x:.9f} {y:.9f} {z:.9f}" size="0.01"/>""")
            continue
        color = "0.85 0.35 0.2 0.5" if s > 0 else "0.2 0.45 0.85 0.5"
        rotor_xml.append(f"""
      <geom name="boom{i}" type="capsule" fromto="0 0 0 {x:.9f} {y:.9f} 0" size="0.008" mass="0.02" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
      <geom name="motor{i}" type="cylinder" pos="{x:.9f} {y:.9f} 0.005" size="0.015 0.015" mass="{cfg.motor_mass}" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
      <geom name="prop{i}" type="cylinder" pos="{x:.9f} {y:.9f} {z + 0.005:.9f}" size="{cfg.prop_radius} 0.002" mass="0" contype="0" conaffinity="0" rgba="{color}"/>
      <site name="rotor{i}" pos="{x:.9f} {y:.9f} {z:.9f}" size="0.01"/>""")

    def link(i, body):
        lo, hi = cfg.joint_ranges[i]
        return f"""
        <joint name="arm_j{i + 1}" type="hinge" axis="0 1 0" range="{lo} {hi}" damping="0.02" armature="0.002"/>
        <geom name="arm_link{i + 1}" type="capsule" fromto="0 0 0 {L[i]} 0 0" size="0.012" mass="{cfg.link_masses[i]}" contype="1" conaffinity="0" rgba="0.75 0.75 0.78 1"/>
        {body}"""

    ee = f"""<body name="ee" pos="{L[2]} 0 0">
              <geom name="ee_tool" type="sphere" size="0.02" mass="{cfg.ee_mass}" contype="1" conaffinity="0" rgba="0.95 0.6 0.1 1"/>
              <site name="ee" size="0.01"/>
              <camera name="ee_cam" pos="0.02 0 0.03" xyaxes="0 -1 0 0 0 1" fovy="90"/>
            </body>"""
    arm = link(0, f"""<body name="arm_l2" pos="{L[0]} 0 0">{link(1, f'''<body name="arm_l3" pos="{L[1]} 0 0">{link(2, ee)}
          </body>''')}
        </body>""")

    spec = None
    if cfg.arm_spec is not None:
        from . import omx_arm
        spec = omx_arm.load_spec(cfg.arm_spec)
        arm = omx_arm.arm_xml(spec)

    actuators = []
    for i, s in enumerate(spins):
        actuators.append(
            f'<general name="thrust{i}" site="rotor{i}" gear="0 0 1 0 0 {-s * cfg.yaw_coef:.5f}" '
            f'dyntype="filter" dynprm="{cfg.motor_tau}" gaintype="fixed" gainprm="1" biastype="none" '
            f'ctrlrange="0 {cfg.thrust_max}" ctrllimited="true"/>')
    for i in range(3):
        lo, hi = cfg.joint_ranges[i]
        actuators.append(
            f'<position name="servo{i + 1}" joint="arm_j{i + 1}" kp="{cfg.servo_kp}" kv="{cfg.servo_kv}" '
            f'ctrlrange="{lo} {hi}" forcerange="{-cfg.servo_tau_max} {cfg.servo_tau_max}"/>')

    if meshes:
        a = 0.065 * np.sin(np.radians(45))      # arm roots on the central board
        r_leg = (cfg.arm_radius - 0.0072) * np.sin(np.radians(45))
        arm_geoms = "".join(
            f'\n      <geom name="frame_arm{k}" type="mesh" mesh="f450_arm" pos="{sx * a:.6f} {sy * a:.6f} -0.012" euler="0 0 {yaw:.6f}" mass="0" contype="0" conaffinity="0" rgba="0.92 0.92 0.9 1"/>'
            for k, (sx, sy, yaw) in enumerate([(1, -1, -np.pi / 4), (1, 1, np.pi / 4), (-1, -1, -3 * np.pi / 4), (-1, 1, 3 * np.pi / 4)]))
        leg_geoms = "".join(
            f'\n      <geom name="leg_vis{k}" type="mesh" mesh="f450_leg_170mm" pos="{sx * r_leg:.6f} {sy * r_leg:.6f} -0.110" euler="0 0 {yaw:.6f}" mass="0" contype="0" conaffinity="0" rgba="{c}"/>'
            f'\n      <geom name="leg{k}" type="capsule" fromto="{sx * r_leg:.6f} {sy * r_leg:.6f} -0.03 {sx * r_leg:.6f} {sy * r_leg:.6f} -0.19" size="0.008" mass="0" contype="1" conaffinity="0" group="3"/>'
            for k, (sx, sy, yaw, c) in enumerate([(1, -1, 3 * np.pi / 4, "0.92 0.92 0.9 1"), (1, 1, -3 * np.pi / 4, "0.92 0.92 0.9 1"),
                                                   (-1, -1, np.pi / 4, "0.85 0.2 0.2 1"), (-1, 1, -np.pi / 4, "0.85 0.2 0.2 1")]))
        base_xml = f"""
      <geom name="base_box" type="box" pos="0 0 -0.03" size="{bx} {by} {bz}" mass="0" contype="1" conaffinity="0" group="3"/>
      <geom name="plate_bottom" type="mesh" mesh="f450_central_bottom" pos="0 0 -0.05" mass="0" contype="0" conaffinity="0" rgba="0.15 0.15 0.17 1"/>
      <geom name="plate_top" type="mesh" mesh="f450_central_top" pos="0 0 -0.012" mass="0" contype="0" conaffinity="0" rgba="0.15 0.15 0.17 1"/>{arm_geoms}{leg_geoms}"""
        mesh_assets = "".join(
            f'\n    <mesh name="f450_{n}" file="{os.path.join(cfg.mesh_dir, "f450_" + n + ".obj")}"/>'
            for n in ("central_bottom", "central_top", "arm", "rotor", "prop", "leg_170mm"))
    else:
        base_xml = f"""
      <geom name="base_box" type="box" size="{bx} {by} {bz}" mass="{cfg.base_mass}" contype="1" conaffinity="0" rgba="0.25 0.25 0.3 1"/>
      <geom name="leg_l" type="capsule" fromto="0 0.08 -{bz} 0 0.12 -0.16" size="0.006" mass="0.02" contype="1" conaffinity="0"/>
      <geom name="leg_r" type="capsule" fromto="0 -0.08 -{bz} 0 -0.12 -0.16" size="0.006" mass="0.02" contype="1" conaffinity="0"/>"""
        mesh_assets = ""
    if spec is not None:
        if cfg.base_inertia is None:
            raise ValueError("the OMX arm needs an explicit base_inertia (use f450_stock / f450_mrs)")
        inertial, mount_geom = omx_arm.base_inertial_with_mount(cfg, spec)
        base_xml = inertial + base_xml + mount_geom
        mesh_assets += omx_arm.mesh_assets(spec)
    elif cfg.base_inertia is not None:
        ixx, iyy, izz = cfg.base_inertia
        base_xml = f"""
      <inertial pos="0 0 0" mass="{cfg.base_mass}" diaginertia="{ixx:.6f} {iyy:.6f} {izz:.6f}"/>""" + base_xml

    return f"""<mujoco model="uam_{cfg.n_rotors}rotor_arm3">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{cfg.timestep}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.68 0.82" rgb2="0.12 0.16 0.22" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".15 .2 .25" rgb2=".25 .3 .35" width="512" height="512" mark="edge" markrgb=".3 .35 .4"/>
    <material name="grid" texture="grid" texrepeat="4 4" texuniform="true" reflectance=".1"/>{mesh_assets}
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="ground" type="plane" size="10 10 0.1" material="grid"/>
    <camera name="side" pos="0 -1.3 1.15" xyaxes="1 0 0 0 0.1 1" fovy="50"/>{extra_world}
    <body name="base" pos="0 0 {cfg.start_height}">
      <freejoint name="base_free"/>
{base_xml}
      <site name="imu" size="0.01"/>{''.join(rotor_xml)}
      <body name="arm_l1" pos="{cfg.arm_mount[0]} {cfg.arm_mount[1]} {cfg.arm_mount[2]}">{arm}
      </body>
    </body>
  </worldbody>
  <actuator>
    {chr(10).join('    ' + a for a in actuators).strip()}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <framepos name="ee_pos" objtype="site" objname="ee"/>
  </sensor>
</mujoco>
"""


def load_model(cfg: UAMConfig | None = None, extra_world: str = ""):
    import mujoco
    cfg = cfg or UAMConfig()
    return mujoco.MjModel.from_xml_string(build_mjcf(cfg, extra_world)), cfg

"""Build the planar 3-DoF arm assets from ROBOTIS OpenMANIPULATOR-X (OMX) source files.

Reproduces meshes/*.obj, arm_spec.json and arm_only.xml in this folder.

Source: github.com/ROBOTIS-GIT/open_manipulator (branch main, commit d31000d9, Apache-2.0),
  open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro
  open_manipulator_description/meshes/open_manipulator_x/*.stl   (millimetres, URDF scale 0.001)

Mapping (OMX -> uam_quad):
  OMX joint2 (shoulder pitch) -> arm_j1, OMX link3 -> arm_l1
  OMX joint3 (elbow pitch)    -> arm_j2, OMX link4 -> arm_l2
  OMX joint4 (wrist pitch)    -> arm_j3, OMX link5 -> arm_l3 (gripper body incl. gripper servo)
  OMX end_effector_link       -> ee
  OMX joint1 (base yaw) is dropped (locked at 0); OMX link2 (shoulder bracket + joint2 servo)
  becomes a static mount; OMX link1 (yaw servo/base) is not used.
All OMX pitch axes are +y, same sign as uam_quad (+q pitches a +x-pointing tip down).

OMX link3 is L-shaped: at q2=0 it points (0.024, 0, 0.128) from joint2 to joint3, i.e. almost
straight up. uam_quad wants zero angle = link along +x, so arm_l1's frame is the OMX link3 frame
rotated by Ry(-theta1) with theta1 = atan2(0.128, 0.024). Meshes, CoM and inertia are re-expressed
in that frame (baked into the OBJ vertices), and joint angles map as
  arm_j1 = q2_omx - theta1,  arm_j2 = q3_omx + theta1,  arm_j3 = q4_omx.
Links 4 and 5 already extend along +x, so their frames are unchanged.

Usage:  source setup/env.sh; python build_arm.py [--omx-root PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/ROBOTIS-GIT/open_manipulator"
COMMIT = "d31000d90c679af9c982e73de8b12d777c5ff7dd"
URDF_PATH = "open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro"
URDF_URL = f"{REPO}/blob/{COMMIT}/{URDF_PATH}"
MESH_URL = f"{REPO}/tree/{COMMIT}/open_manipulator_description/meshes/open_manipulator_x"
XM430_URL = "https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/"
OMX_SPEC_URL = "https://emanual.robotis.com/docs/en/platform/openmanipulator_x/specification/"

# ---- Values copied verbatim from open_manipulator_x_arm.urdf.xacro (commit above) ----
OMX = {
    "link2": dict(mass=9.8406837e-02, com=(-3.0184870e-04, 5.4043684e-04, 0.018 + 2.9433464e-02),
                  I=(3.4543422e-05, 3.2689329e-05, 1.8850320e-05, -1.6031095e-08, -3.8375155e-07, 2.8511935e-08),
                  visual_xyz=(0, 0, 0.019)),
    "link3": dict(mass=1.3850917e-01, com=(1.0308393e-02, 3.7743363e-04, 1.0170197e-01),
                  I=(3.3055381e-04, 3.4290447e-04, 6.0346498e-05, -9.7940978e-08, -3.8505711e-05, -1.5717516e-06),
                  visual_xyz=(0, 0, 0)),
    "link4": dict(mass=1.3274562e-01, com=(9.0909590e-02, 3.8929816e-04, 2.2413279e-04),
                  I=(3.0654178e-05, 2.4230292e-04, 2.5155057e-04, -1.2764155e-06, -2.6874417e-07, 1.1559550e-08),
                  visual_xyz=(0, 0, 0)),
    "link5": dict(mass=1.4327573e-01, com=(4.4206755e-02, 3.6839985e-07, 8.9142216e-03),
                  I=(8.0870749e-05, 7.5980465e-05, 9.3127351e-05, 0.0, -1.0157896e-06, 0.0),
                  visual_xyz=(0, 0, 0)),
    # finger palms: URDF placeholder inertials (1 g, 1e-6 kg m^2 diag)
    "gripper_left_palm": dict(mass=1.0e-03, com=(0, 0, 0), I=(1e-6, 1e-6, 1e-6, 0, 0, 0)),
    "gripper_right_palm": dict(mass=1.0e-03, com=(0, 0, 0), I=(1e-6, 1e-6, 1e-6, 0, 0, 0)),
}
J2_IN_LINK2 = (0.0, 0.0, 0.0595)        # joint2 origin
J3_IN_LINK3 = (0.024, 0.0, 0.128)       # joint3 origin
J4_IN_LINK4 = (0.124, 0.0, 0.0)         # joint4 origin
EE_IN_LINK5 = (0.126, 0.0, 0.0)         # end_effector_joint origin
FINGER_L_IN_LINK5 = (0.0817, 0.021, 0.0)
FINGER_R_IN_LINK5 = (0.0817, -0.021, 0.0)
LIMITS_OMX = {"joint2": (-1.5, 1.5), "joint3": (-1.5, 1.4), "joint4": (-1.7, 1.97)}  # main branch
LIMITS_OMX_NOETIC = {"joint2": (-math.pi * 0.57, math.pi * 0.5), "joint3": (-math.pi * 0.3, math.pi * 0.44),
                     "joint4": (-math.pi * 0.57, math.pi * 0.65)}
URDF_VEL = 4.8                           # rad/s, URDF <limit velocity>

# ---- XM430-W350-T/R datasheet (ROBOTIS e-Manual) ----
XM430 = dict(model="DYNAMIXEL XM430-W350-T", mass_kg=0.082, gear_ratio=353.5,
             stall_torque_Nm={"11.1V": 3.8, "12.0V": 4.1, "14.8V": 4.8},
             no_load_speed_rpm={"11.1V": 43, "12.0V": 46, "14.8V": 57},
             voltage_range_V=[10.0, 14.8], resolution_pulse_per_rev=4096, source=XM430_URL)

THETA1 = math.atan2(J3_IN_LINK3[2], J3_IN_LINK3[0])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def full_I(ixx, iyy, izz, ixy, ixz, iyz):
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def mj_fullinertia(I):
    return [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]


def main(omx_root):
    mesh_src = os.path.join(omx_root, "open_manipulator_description/meshes/open_manipulator_x")
    out_mesh = os.path.join(HERE, "meshes")
    os.makedirs(out_mesh, exist_ok=True)

    # R_new: columns = new link axes expressed in the OMX link frame. p_new = R_new.T @ p_omx.
    frames = {"link2": (Ry(0.0), np.array(J2_IN_LINK2)),   # mount: origin at joint2 axis
              "link3": (Ry(-THETA1), np.zeros(3)),
              "link4": (Ry(0.0), np.zeros(3)),
              "link5": (Ry(0.0), np.zeros(3)),
              "gripper_left_palm": (Ry(0.0), np.zeros(3)),
              "gripper_right_palm": (Ry(0.0), np.zeros(3))}
    names = {"link2": "arm_mount", "link3": "arm_l1", "link4": "arm_l2", "link5": "arm_l3",
             "gripper_left_palm": "finger_left", "gripper_right_palm": "finger_right"}

    out = {}
    for omx_name, (R, origin) in frames.items():
        p = OMX[omx_name]
        m = trimesh.load(os.path.join(mesh_src, omx_name + ".stl"))
        v = m.vertices * 0.001 + np.array(p.get("visual_xyz", (0, 0, 0)))  # mm -> m, URDF visual origin
        m.vertices = (v - origin) @ R                                          # (R.T @ (v - o)) row-wise
        fname = f"{names[omx_name]}.obj"
        m.export(os.path.join(out_mesh, fname))
        com = R.T @ (np.array(p["com"]) - origin)
        I = R.T @ full_I(*p["I"]) @ R
        ev = np.linalg.eigvalsh(I)
        out[omx_name] = dict(name=names[omx_name], mesh=fname, mass=p["mass"], com=com, I=I,
                             bounds=m.bounds.round(4).tolist(), principal=ev)

    L1 = float(np.hypot(J3_IN_LINK3[0], J3_IN_LINK3[2]))
    L2, L3 = J4_IN_LINK4[0], EE_IN_LINK5[0]
    lim = {"arm_j1": [LIMITS_OMX["joint2"][0] - THETA1, LIMITS_OMX["joint2"][1] - THETA1],
           "arm_j2": [LIMITS_OMX["joint3"][0] + THETA1, LIMITS_OMX["joint3"][1] + THETA1],
           "arm_j3": list(LIMITS_OMX["joint4"])}

    def r(x, n=9):
        return [round(float(a), n) for a in np.ravel(x)]

    links = []
    for i, (omx_name, joint, omx_joint, L, nxt) in enumerate([
            ("link3", "arm_j1", "joint2", L1, "arm_l2"),
            ("link4", "arm_j2", "joint3", L2, "arm_l3"),
            ("link5", "arm_j3", "joint4", L3, "ee")]):
        o = out[omx_name]
        links.append({
            "body": o["name"], "source_link": f"OMX {omx_name}", "joint": joint, "source_joint": f"OMX {omx_joint}",
            "joint_axis": [0, 1, 0], "joint_pos_in_parent": None,  # filled below
            "length_to_next_joint_m": round(L, 6), "next_frame": nxt, "next_frame_pos": [round(L, 6), 0, 0],
            "mass_kg": o["mass"], "com_m": r(o["com"]),
            "inertia_about_com_kg_m2": {"fullinertia_xx_yy_zz_xy_xz_yz": r(mj_fullinertia(o["I"]), 12),
                                        "principal_moments": r(o["principal"], 12),
                                        "frame": "link frame (x toward next joint, y = joint axis), about CoM"},
            "range_rad": r(lim[joint], 6),
            "range_source_rad": list(LIMITS_OMX[omx_joint]),
            "range_source_noetic_rad": r(LIMITS_OMX_NOETIC[omx_joint], 6),
            "angle_map": {"arm_j1": "q_uam = q_omx_joint2 - theta1", "arm_j2": "q_uam = q_omx_joint3 + theta1",
                          "arm_j3": "q_uam = q_omx_joint4"}[joint],
            "servo": "XM430-W350-T" + (" (servo lives in OMX link2 = mount)" if i == 0 else
                                       f" (servo lives in OMX link{i + 2} = {links[i - 1]['body'] if i else ''})"),
            "max_velocity_rad_s": URDF_VEL,
            "mesh": {"file": f"meshes/{o['mesh']}", "pos": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0],
                     "note": "vertices already expressed in this link frame (meters)",
                     "bounds_m": o["bounds"]},
        })
    links[0]["joint_pos_in_parent"] = "arm mount point (joint-2 axis of OMX), set by the airframe"
    links[1]["joint_pos_in_parent"] = [round(L1, 6), 0, 0]
    links[2]["joint_pos_in_parent"] = [L2, 0, 0]
    links[0]["servo"] = "XM430-W350-T in OMX link2 (mount, on the airframe) drives arm_j1"
    links[1]["servo"] = "XM430-W350-T inside arm_l1 (OMX link3) drives arm_j2"
    links[2]["servo"] = "XM430-W350-T inside arm_l2 (OMX link4) drives arm_j3; arm_l3 carries the gripper XM430"

    fl, fr = out["gripper_left_palm"], out["gripper_right_palm"]
    mount = out["link2"]
    arm_mass = sum(OMX[k]["mass"] for k in ("link3", "link4", "link5", "gripper_left_palm", "gripper_right_palm"))

    spec = {
        "name": "planar 3-DoF pitch arm from ROBOTIS OpenMANIPULATOR-X (RM-X52-TNM)",
        "license": "Apache-2.0 (Copyright ROBOTIS CO., LTD.); see README.md",
        "sources": {"repo": REPO, "commit": COMMIT, "urdf": URDF_URL, "meshes": MESH_URL,
                    "servo_datasheet": XM430_URL, "omx_spec": OMX_SPEC_URL},
        "convention": "joints rotate about +y; zero angle = link along +x; +q pitches the tip down (toward -z). "
                      "Each link frame origin is on its joint axis.",
        "theta1_rad": round(THETA1, 9),
        "theta1_note": "OMX link3 is L-shaped (joint2->joint3 = (0.024, 0, 0.128) m); arm_l1 frame = OMX link3 frame "
                       "rotated by Ry(-theta1) so joint3 lies on +x. OMX home pose (all zero) = "
                       "arm_j1 = -theta1 (upper arm ~up), arm_j2 = +theta1, arm_j3 = 0.",
        "links": links,
        "ee": {"body": "ee", "site": "ee", "parent": "arm_l3", "pos": [L3, 0, 0],
               "source": "OMX end_effector_joint origin (0.126, 0, 0) in link5 (between the fingertips)"},
        "fingers": [{"body": "finger_left", "parent": "arm_l3", "pos": list(FINGER_L_IN_LINK5),
                     "mesh": "meshes/finger_left.obj", "mass_kg": 1e-3, "inertia_diag": [1e-6] * 3,
                     "note": "OMX prismatic gripper_left_joint frozen at 0; URDF placeholder inertial"},
                    {"body": "finger_right", "parent": "arm_l3", "pos": list(FINGER_R_IN_LINK5),
                     "mesh": "meshes/finger_right.obj", "mass_kg": 1e-3, "inertia_diag": [1e-6] * 3,
                     "note": "OMX gripper_right_joint (mimic) frozen at 0; URDF placeholder inertial"}],
        "mount": {"body": "arm_mount", "source_link": "OMX link2 (joint2 servo + shoulder bracket; yaw locked)",
                  "mesh": "meshes/arm_mount.obj", "mesh_pose": "vertices in a frame whose origin is the arm_j1 axis",
                  "mass_kg": mount["mass"], "com_m": r(mount["com"]),
                  "inertia_fullinertia": r(mj_fullinertia(mount["I"]), 12),
                  "bounds_m": mount["bounds"],
                  "note": "belongs to the airframe (fixed). OMX link1 (yaw servo, 0.0791 kg) not included."},
        "moving_mass_kg": round(arm_mass, 9),
        "moving_mass_note": "arm_l1 + arm_l2 + arm_l3 + 2 finger palms; excludes mount",
        "servo": XM430,
        "servo_in_sim": {"type": "MuJoCo position", "kp": 20.0, "kv": 0.6, "forcerange_Nm": 4.1,
                         "note": "kp/kv kept from the placeholder (not identified); force limit = XM430-W350 "
                                 "stall torque at 12 V. At 4S (14.8 V) stall is 4.8 N m."},
        "omx_official": {"weight_kg": 0.70, "payload_kg": 0.5, "reach_m": 0.38, "joint_speed_rpm": 46,
                         "source": OMX_SPEC_URL},
    }
    with open(os.path.join(HERE, "arm_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # ---------------- standalone test MJCF ----------------
    def inertial(o):
        c = " ".join(f"{a:.9g}" for a in o["com"])
        fi = " ".join(f"{a:.9g}" for a in mj_fullinertia(o["I"]))
        return f'<inertial pos="{c}" mass="{o["mass"]:.9g}" fullinertia="{fi}"/>'

    l1, l2, l3 = out["link3"], out["link4"], out["link5"]
    vis = 'contype="0" conaffinity="0" group="2"'
    col = 'contype="1" conaffinity="0" group="3"'
    rng = {k: f"{v[0]:.6f} {v[1]:.6f}" for k, v in lim.items()}
    xml = f"""<mujoco model="omx_planar_3dof">
  <!-- Generated by build_arm.py from ROBOTIS OpenMANIPULATOR-X (Apache-2.0). Fixed-base test model. -->
  <compiler angle="radian" autolimits="true" meshdir="meshes"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <default>
    <geom type="mesh" mass="0"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .75" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"/>
    <material name="omx_dark" rgba="0.2 0.2 0.2 1"/>
    <material name="omx_l1" rgba="0.25 0.45 0.8 1"/>
    <material name="omx_l2" rgba="0.8 0.45 0.2 1"/>
    <material name="omx_l3" rgba="0.3 0.65 0.35 1"/>
    <mesh name="arm_mount" file="arm_mount.obj"/>
    <mesh name="arm_l1" file="arm_l1.obj"/>
    <mesh name="arm_l2" file="arm_l2.obj"/>
    <mesh name="arm_l3" file="arm_l3.obj"/>
    <mesh name="finger_left" file="finger_left.obj"/>
    <mesh name="finger_right" file="finger_right.obj"/>
  </asset>
  <worldbody>
    <light pos="0 -0.5 1.5" dir="0 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="1 1 0.01" material="grid" pos="0 0 -0.3" mass="0"/>
    <camera name="side" pos="0.17 -0.75 0.05" xyaxes="1 0 0 0 0 1" fovy="45"/>
    <camera name="iso" pos="0.55 -0.55 0.35" xyaxes="0.707 0.707 0 -0.3 0.3 0.905" fovy="45"/>
    <body name="arm_mount" pos="0 0 0">
      {inertial(mount)}
      <geom name="arm_mount_vis" mesh="arm_mount" material="omx_dark" {vis}/>
      <body name="arm_l1" pos="0 0 0">
        <joint name="arm_j1" type="hinge" axis="0 1 0" range="{rng['arm_j1']}" damping="0.02" armature="0.002"/>
        {inertial(l1)}
        <geom name="arm_link1" mesh="arm_l1" material="omx_l1" {vis}/>
        <geom name="arm_link1_col" mesh="arm_l1" {col}/>
        <body name="arm_l2" pos="{L1:.9f} 0 0">
          <joint name="arm_j2" type="hinge" axis="0 1 0" range="{rng['arm_j2']}" damping="0.02" armature="0.002"/>
          {inertial(l2)}
          <geom name="arm_link2" mesh="arm_l2" material="omx_l2" {vis}/>
          <geom name="arm_link2_col" mesh="arm_l2" {col}/>
          <body name="arm_l3" pos="{L2:.9f} 0 0">
            <joint name="arm_j3" type="hinge" axis="0 1 0" range="{rng['arm_j3']}" damping="0.02" armature="0.002"/>
            {inertial(l3)}
            <geom name="arm_link3" mesh="arm_l3" material="omx_l3" {vis}/>
            <geom name="arm_link3_col" mesh="arm_l3" {col}/>
            <body name="finger_left" pos="{FINGER_L_IN_LINK5[0]} {FINGER_L_IN_LINK5[1]} {FINGER_L_IN_LINK5[2]}">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
              <geom name="finger_left_vis" mesh="finger_left" material="omx_dark" {vis}/>
            </body>
            <body name="finger_right" pos="{FINGER_R_IN_LINK5[0]} {FINGER_R_IN_LINK5[1]} {FINGER_R_IN_LINK5[2]}">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
              <geom name="finger_right_vis" mesh="finger_right" material="omx_dark" {vis}/>
            </body>
            <body name="ee" pos="{L3} 0 0">
              <site name="ee" size="0.006" rgba="1 0 0 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="servo1" joint="arm_j1" kp="20" kv="0.6" ctrlrange="{rng['arm_j1']}" forcerange="-4.1 4.1"/>
    <position name="servo2" joint="arm_j2" kp="20" kv="0.6" ctrlrange="{rng['arm_j2']}" forcerange="-4.1 4.1"/>
    <position name="servo3" joint="arm_j3" kp="20" kv="0.6" ctrlrange="{rng['arm_j3']}" forcerange="-4.1 4.1"/>
  </actuator>
</mujoco>
"""
    with open(os.path.join(HERE, "arm_only.xml"), "w") as f:
        f.write(xml)
    print(json.dumps({k: (v["name"], v["bounds"]) for k, v in out.items()}, indent=1))
    print("L =", L1, L2, L3, " theta1 =", THETA1, " moving mass =", arm_mass, " limits =", lim)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--omx-root", default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../external/open_manipulator")))
    main(ap.parse_args().omx_root)

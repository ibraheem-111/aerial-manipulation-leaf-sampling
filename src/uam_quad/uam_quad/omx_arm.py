"""ROBOTIS OpenMANIPULATOR-X, planar 3-DoF version (assets/arm, Apache-2.0), for the UAM model.

The three parallel pitch joints of the OMX (its joints 2-4, DYNAMIXEL XM430-W350-T) with CAD
masses, CoMs and full inertia tensors from the official URDF. The shoulder bracket (OMX link2) is
rigidly attached to the airframe, so its mass properties are merged into the base inertial.
"""
from __future__ import annotations

import json
import os

import numpy as np

DEFAULT_ARM_SPEC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "arm", "arm_spec.json")
COLORS = ["0.25 0.45 0.8 1", "0.8 0.45 0.2 1", "0.3 0.65 0.35 1"]
FINGER_POS = {"left": (0.0817, 0.021, 0.0), "right": (0.0817, -0.021, 0.0)}


def load_spec(path=DEFAULT_ARM_SPEC):
    spec = json.load(open(path))
    spec["_dir"] = os.path.dirname(os.path.abspath(path))
    return spec


def apply_to_config(cfg, spec_path=DEFAULT_ARM_SPEC):
    spec = load_spec(spec_path)
    L = spec["links"]
    cfg.arm_spec = spec_path
    cfg.link_lengths = (L[0]["length_to_next_joint_m"], L[1]["length_to_next_joint_m"], spec["ee"]["pos"][0])
    cfg.link_masses = tuple(l["mass_kg"] for l in L)
    cfg.ee_mass = 0.0
    cfg.joint_ranges = tuple(tuple(l["range_rad"]) for l in L)
    cfg.servo_tau_max = spec["servo"]["stall_torque_Nm"]["12.0V"]
    return cfg


def full_inertia(v):
    xx, yy, zz, xy, xz, yz = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def combine(m1, c1, I1, m2, c2, I2):
    """Mass, CoM and inertia about the combined CoM (inputs: inertias about each body's own CoM)."""
    m = m1 + m2
    c = (m1 * np.asarray(c1) + m2 * np.asarray(c2)) / m

    def shift(mk, ck):
        d = np.asarray(ck) - c
        return mk * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, I1 + shift(m1, c1) + I2 + shift(m2, c2)


def mesh_assets(spec):
    d = os.path.join(spec["_dir"], "meshes")
    return "".join(f'\n    <mesh name="omx_{n}" file="{os.path.join(d, n + ".obj")}"/>'
                   for n in ("arm_mount", "arm_l1", "arm_l2", "arm_l3", "finger_left", "finger_right"))


def _link(spec, i, inner):
    l = spec["links"][i]
    I = " ".join(str(v) for v in l["inertia_about_com_kg_m2"]["fullinertia_xx_yy_zz_xy_xz_yz"])
    lo, hi = l["range_rad"]
    c = l["com_m"]
    return (f'\n        <joint name="arm_j{i + 1}" type="hinge" axis="0 1 0" range="{lo} {hi}" damping="0.02" armature="0.002"/>'
            f'\n        <inertial pos="{c[0]} {c[1]} {c[2]}" mass="{l["mass_kg"]}" fullinertia="{I}"/>'
            f'\n        <geom name="arm_link{i + 1}" type="mesh" mesh="omx_arm_l{i + 1}" mass="0" contype="0" conaffinity="0" rgba="{COLORS[i]}"/>'
            f'\n        <geom name="arm_link{i + 1}_col" type="mesh" mesh="omx_arm_l{i + 1}" mass="0" contype="1" conaffinity="0" group="3"/>'
            + inner)


def arm_xml(spec):
    """Body XML for the contents of arm_l1 (joint + inertial + geoms + children), like model.build_mjcf."""
    L = spec["links"]
    fingers = "".join(
        f'\n          <geom name="finger_{s}" type="mesh" mesh="omx_finger_{s}" pos="{p[0]} {p[1]} {p[2]}" mass="0" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>'
        for s, p in FINGER_POS.items())
    ee = (fingers + f'\n          <body name="ee" pos="{spec["ee"]["pos"][0]} 0 0">'
          '\n            <site name="ee" size="0.008" rgba="1 0.2 0.2 1"/>'
          '\n            <geom name="tip" type="sphere" size="0.008" mass="0" contype="1" conaffinity="0" rgba="0.95 0.6 0.1 1"/>'
          '\n            <camera name="ee_cam" pos="-0.03 0 0.04" xyaxes="0 -1 0 0.6 0 0.8" fovy="90"/>'
          '\n          </body>')
    l3 = f'\n        <body name="arm_l3" pos="{L[1]["length_to_next_joint_m"]} 0 0">' + _link(spec, 2, ee) + '\n        </body>'
    l2 = f'\n        <body name="arm_l2" pos="{L[0]["length_to_next_joint_m"]} 0 0">' + _link(spec, 1, l3) + '\n        </body>'
    return _link(spec, 0, l2)


def base_inertial_with_mount(cfg, spec):
    """Inertial XML for the base body = airframe (cfg.base_mass, cfg.base_inertia at O) + shoulder bracket."""
    m = spec["mount"]
    c_m = np.asarray(cfg.arm_mount, dtype=float) + np.asarray(m["com_m"])
    mb, cb, Ib = combine(cfg.base_mass, np.zeros(3), np.diag(cfg.base_inertia), m["mass_kg"], c_m, full_inertia(m["inertia_fullinertia"]))
    inertial = (f'\n      <inertial pos="{cb[0]:.9f} {cb[1]:.9f} {cb[2]:.9f}" mass="{mb:.9f}" '
                f'fullinertia="{Ib[0, 0]:.9e} {Ib[1, 1]:.9e} {Ib[2, 2]:.9e} {Ib[0, 1]:.9e} {Ib[0, 2]:.9e} {Ib[1, 2]:.9e}"/>')
    a = cfg.arm_mount
    geom = (f'\n      <geom name="arm_mount" type="mesh" mesh="omx_arm_mount" pos="{a[0]} {a[1]} {a[2]}" '
            'mass="0" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>')
    return inertial, geom

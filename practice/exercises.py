"""MuJoCo practice test -- fill in each function, then run:  python check.py

Work top to bottom; each exercise builds a skill you'll use live (load, inspect, step, sense, act,
contact, render). Useful references:
  * MuJoCo Python docs: https://mujoco.readthedocs.io/en/stable/python.html
  * API (mj_step, mj_forward, mj_contactForce, ...): https://mujoco.readthedocs.io/en/stable/APIreference/index.html
  * Our package: src/uam_quad (uam_quad.load_model, hover_trim, GeometricController, ...)
Answers are in solutions.py -- try first.
"""
import os

import mujoco  # noqa: F401
import numpy as np  # noqa: F401

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "uam_quad", "models")
F450_XML = os.path.join(MODELS, "f450_omx.xml")

# A simple pendulum: 0.2 kg ball on a massless 0.5 m rod, hinge about y at the origin.
PENDULUM_XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.005" mass="0"/>
      <geom name="bob" type="sphere" pos="0 0 -0.5" size="0.02" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>"""

# A 1 kg box resting on the floor (for the contact exercise).
BOX_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.2">
      <freejoint/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>"""


def ex01_load_pendulum():
    """Return an MjModel built from PENDULUM_XML.  (hint: mujoco.MjModel.from_xml_string)"""
    raise NotImplementedError


def ex02_sizes():
    """Load F450_XML from file and return (nq, nv, nu, nbody).
    Why is nq != nv?  (write your answer in a comment)"""
    raise NotImplementedError


def ex03_pendulum_period():
    """Start the pendulum at 0.1 rad, simulate 10 s with mj_step, and return the oscillation
    period in seconds (measure it from zero crossings of the joint angle data.qpos[0])."""
    raise NotImplementedError


def ex04_total_mass():
    """Return the total mass [kg] of the drone + arm in F450_XML.
    (hint: model.body_subtreemass for the body named 'base')"""
    raise NotImplementedError


def ex05_actuator_names():
    """Return the list of actuator names in F450_XML, in order.
    Which ones are rotors, which are arm servos?"""
    raise NotImplementedError


def ex06_hover_thrusts():
    """Return the 4 rotor thrusts [N] that hold the stock F450 + OpenMANIPULATOR-X still in hover
    with the arm at (-0.6, 1.0, 0.4).  (hint: uam_quad.hover_trim, f450_stock, with_omx_arm, load_model)
    Why are the front two larger than the rear two?"""
    raise NotImplementedError


def ex07_read_sensor():
    """Load F450_XML, call mj_forward, and return the value of the sensor named 'ee_pos'
    (the end-effector position).  (hint: data.sensor('ee_pos').data)"""
    raise NotImplementedError


def ex08_free_fall():
    """Load F450_XML, lift the drone to 3 m (data.qpos[2] = 3.0), leave all controls at 0 (motors off),
    simulate 0.5 s, and return how far it fell [m].  Compare with 0.5 g t^2.
    (Why lift it? From the default 1 m it hits the ground after ~0.8 m -- try it and look.)"""
    raise NotImplementedError


def ex09_arm_pose_effect():
    """Return (front_thrust_arm_out, front_thrust_arm_folded): the largest hover rotor thrust with the
    arm at (-0.6, 1.0, 0.4) and with it folded at (-2.0, 2.4, 0.0).  Explain the difference."""
    raise NotImplementedError


def ex10_hover_controller():
    """Use uam_quad.GeometricController to hold the drone at (0, 0, 1) starting 10 cm to the side
    (x = 0.10).  Simulate 4 s (servos at the trim setpoints) and return the final position error [m]."""
    raise NotImplementedError


def ex11_contact_force():
    """Load BOX_XML, simulate 1 s so the box settles, and return the total normal contact force [N]
    between the box and the floor.  (hint: loop over data.contact, mujoco.mj_contactForce)
    What value do you expect, and why?"""
    raise NotImplementedError


def ex12_render_png(path):
    """Render the F450 from an offscreen camera and save it as a PNG at `path`.
    (hint: mujoco.Renderer, renderer.update_scene(data, camera=...), PIL.Image.fromarray)"""
    raise NotImplementedError

"""Worked solutions for exercises.py (python check.py --solutions)."""
import mujoco
import numpy as np

from exercises import BOX_XML, F450_XML, PENDULUM_XML

TH = np.array([-0.6, 1.0, 0.4])


def ex01_load_pendulum():
    return mujoco.MjModel.from_xml_string(PENDULUM_XML)


def ex02_sizes():
    m = mujoco.MjModel.from_xml_path(F450_XML)
    # nq = 10 (free joint: 3 position + 4 quaternion, plus 3 arm joints)
    # nv = 9  (free joint: 3 linear + 3 angular velocities, plus 3 arm joint rates)
    # A quaternion has 4 numbers but only 3 rotational degrees of freedom, so nq > nv.
    return m.nq, m.nv, m.nu, m.nbody


def ex03_pendulum_period():
    m = ex01_load_pendulum()
    d = mujoco.MjData(m)
    d.qpos[0] = 0.1
    t, q = [], []
    while d.time < 10.0:
        mujoco.mj_step(m, d)
        t.append(d.time)
        q.append(d.qpos[0])
    q = np.array(q)
    t = np.array(t)
    up = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]      # upward zero crossings
    return float(np.mean(np.diff(t[up])))              # ~ 2*pi*sqrt(0.5/9.81) = 1.42 s


def ex04_total_mass():
    m = mujoco.MjModel.from_xml_path(F450_XML)
    return float(m.body_subtreemass[m.body("base").id])


def ex05_actuator_names():
    m = mujoco.MjModel.from_xml_path(F450_XML)
    return [m.actuator(i).name for i in range(m.nu)]   # thrust0-3 = rotors, servo1-3 = arm


def _f450():
    from uam_quad import f450_stock, load_model, with_omx_arm
    return load_model(with_omx_arm(f450_stock()))


def ex06_hover_thrusts():
    from uam_quad import hover_trim
    model, cfg = _f450()
    # Front rotors carry more because the arm reaching forward shifts the CoM ahead of the rotor centre.
    return hover_trim(model, cfg, TH).ctrl[:4]


def ex07_read_sensor():
    m = mujoco.MjModel.from_xml_path(F450_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return np.array(d.sensor("ee_pos").data)


def ex08_free_fall():
    m = mujoco.MjModel.from_xml_path(F450_XML)
    d = mujoco.MjData(m)
    d.qpos[2] = 3.0                                    # from 1 m the legs hit the ground after ~0.8 m
    z0 = d.qpos[2]
    while d.time < 0.5 - 1e-9:
        mujoco.mj_step(m, d)
    return float(z0 - d.qpos[2])                       # ~ 0.5 * 9.81 * 0.25 = 1.23 m


def ex09_arm_pose_effect():
    from uam_quad import hover_trim
    model, cfg = _f450()
    out = hover_trim(model, cfg, TH).ctrl[:4].max()
    folded = hover_trim(model, cfg, [-2.0, 2.4, 0.0]).ctrl[:4].max()
    # Folded back over the body, the arm's CoM sits nearer the centre -> thrusts more even.
    return float(out), float(folded)


def ex10_hover_controller():
    from uam_quad import GeometricController, hover_trim
    model, cfg = _f450()
    trim = hover_trim(model, cfg, TH)
    d = mujoco.MjData(model)
    d.qpos[:], d.ctrl[:] = trim.qpos, trim.ctrl
    d.act[:4] = trim.act
    d.qpos[0] += 0.10
    mujoco.mj_forward(model, d)
    ctrl = GeometricController(model, cfg)
    target = np.array([0.0, 0.0, 1.0])
    while d.time < 4.0:
        thrusts, _ = ctrl(d, target)
        d.ctrl[:4] = thrusts
        mujoco.mj_step(model, d)
    return float(np.linalg.norm(d.qpos[0:3] - target))


def ex11_contact_force():
    m = mujoco.MjModel.from_xml_string(BOX_XML)
    d = mujoco.MjData(m)
    while d.time < 1.0:
        mujoco.mj_step(m, d)
    total, f6 = 0.0, np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f6)
        total += f6[0]                                 # f6[0] = normal component in the contact frame
    return total                                       # ~ m g = 9.81 N (it holds the box up)


def ex12_render_png(path):
    from PIL import Image
    m = mujoco.MjModel.from_xml_path(F450_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.qpos[0:3]
    cam.distance, cam.azimuth, cam.elevation = 1.2, -130, -20
    with mujoco.Renderer(m, 480, 640) as r:
        r.update_scene(d, camera=cam)
        Image.fromarray(r.render()).save(path)
    return path

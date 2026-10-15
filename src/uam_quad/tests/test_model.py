import mujoco
import numpy as np
import pytest

from uam_quad import UAMConfig, hover_trim, linearize, controllability_rank, load_model

ARM_POSE = np.array([-0.6, 1.0, 0.4])


@pytest.mark.parametrize("n", [4, 6])
def test_builds_and_counts(n):
    model, cfg = load_model(UAMConfig(n_rotors=n))
    assert model.nu == n + 3
    assert model.na == n                      # one filter state per rotor
    assert model.nv == 6 + 3


def test_free_fall_without_thrust():
    model, _ = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.qacc[2] == pytest.approx(-9.81, abs=1e-3)


def test_allocation_equal_thrust_gives_no_moment():
    for n in (4, 6):
        A = UAMConfig(n_rotors=n).allocation_matrix()
        w = A @ np.ones(n)
        assert w[0] == pytest.approx(n)
        assert np.allclose(w[1:], 0.0, atol=1e-12)
        assert np.linalg.matrix_rank(A) == 4  # upright rotors: only 4 of 6 wrench axes


@pytest.mark.parametrize("n", [4, 6])
def test_hover_trim_holds_still(n):
    model, cfg = load_model(UAMConfig(n_rotors=n))
    trim = hover_trim(model, cfg, ARM_POSE)
    total_mass = model.body_subtreemass[model.body("base").id]
    assert trim.ctrl[:n].sum() == pytest.approx(total_mass * 9.81, rel=1e-6)
    assert trim.residual < 1e-9
    assert trim.qacc_norm < 1e-6


def test_hover_linearization_is_controllable():
    model, cfg = load_model()
    A, B = linearize(model, hover_trim(model, cfg, ARM_POSE))
    # Position/yaw at hover are pure integrators (|eig| = 1) — nothing should be unstable beyond that.
    assert np.max(np.abs(np.linalg.eigvals(A))) < 1.0 + 1e-4
    assert controllability_rank(A, B) == A.shape[0]


@pytest.mark.parametrize("n", [4, 6])
def test_hand_derived_eom_matches_mujoco(n):
    from uam_quad.eom import check_against_mujoco
    model, cfg = load_model(UAMConfig(n_rotors=n))
    lin, ang = check_against_mujoco(model, cfg, trials=30)
    assert lin < 1e-3 and ang < 1e-3  # limited by finite differences, not by the model


def test_dropping_arm_reaction_breaks_eom():
    from uam_quad.eom import check_against_mujoco
    model, cfg = load_model()
    _, ang = check_against_mujoco(model, cfg, trials=10, drop=("arm_reaction",))
    assert ang > 1.0


def test_f450_mrs_preset():
    from uam_quad import f450_mrs
    from uam_quad.eom import check_against_mujoco
    cfg = f450_mrs()
    model, _ = load_model(cfg)
    assert model.nmesh == 6
    assert model.body_subtreemass[model.body("base").id] == pytest.approx(1.704 + 0.38, abs=1e-6)
    A = cfg.allocation_matrix()
    assert np.allclose(A @ np.ones(4), [4, 0, 0, 0], atol=1e-12)  # PX4 order still balanced
    trim = hover_trim(model, cfg, ARM_POSE)
    assert trim.qacc_norm < 1e-6 and trim.ctrl[:4].max() < cfg.thrust_max
    lin, ang = check_against_mujoco(model, cfg, trials=10)
    assert lin < 1e-3 and ang < 1e-3


@pytest.mark.parametrize("omx", [False, True])
def test_wbmpc_prediction_model_matches_mujoco(omx):
    """The MPC's symbolic coupled dynamics and EE kinematics agree with MuJoCo (no ACADOS build needed)."""
    import mujoco
    from uam_quad import f450_stock, with_omx_arm
    from uam_quad.wbmpc import MPCParams, WholeBodyModel, state_from_mujoco
    cfg = f450_stock()
    if omx:
        cfg = with_omx_arm(cfg)
    model, _ = load_model(cfg)
    wb = WholeBodyModel(model, cfg, MPCParams())
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    qv = [model.jnt_dofadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    rng = np.random.default_rng(0)
    data = mujoco.MjData(model)
    for _ in range(20):
        q = rng.normal(size=4)
        data.qpos[:] = 0
        data.qpos[0:3] = [0, 0, 3]
        data.qpos[3:7] = q / np.linalg.norm(q)
        data.qpos[qa] = [rng.uniform(lo + 0.05, hi - 0.05) for lo, hi in cfg.joint_ranges]
        data.qvel[:] = rng.normal(scale=[1, 1, 1, 2, 2, 2, 3, 3, 3])
        data.act[:4] = rng.uniform(0, cfg.thrust_max, 4)
        data.ctrl[:4] = data.act[:4]
        data.ctrl[4:] = data.qpos[qa]
        mujoco.mj_forward(model, data)
        x = state_from_mujoco(model, data, qa, qv)
        pdd, wd = wb.base_acc(x, data.act[:4], data.qacc[qv], np.zeros(3), np.zeros(3))
        assert np.allclose(np.asarray(pdd).ravel(), data.qacc[0:3], atol=1e-6)
        assert np.allclose(np.asarray(wd).ravel(), data.qacc[3:6], atol=1e-5)
        assert np.allclose(np.asarray(wb.f_ee(x)).ravel(), data.site_xpos[model.site("ee").id], atol=1e-9)


def test_omx_arm_mass_properties():
    from uam_quad import f450_stock, with_omx_arm
    model, cfg = load_model(with_omx_arm(f450_stock()))
    assert model.body_subtreemass[model.body("arm_l1").id] == pytest.approx(0.13850917 + 0.13274562 + 0.14327573, abs=1e-6)
    assert hover_trim(model, cfg, ARM_POSE).qacc_norm < 1e-6


def test_wbmpc_model_external_ee_force():
    """A force applied at the end effector in MuJoCo matches the model's f_ext term."""
    import mujoco
    from uam_quad import f450_stock, with_omx_arm
    from uam_quad.wbmpc import MPCParams, WholeBodyModel, state_from_mujoco
    cfg = with_omx_arm(f450_stock())
    model, _ = load_model(cfg)
    wb = WholeBodyModel(model, cfg, MPCParams())
    qa = [model.jnt_qposadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    qv = [model.jnt_dofadr[model.joint(f"arm_j{i}").id] for i in (1, 2, 3)]
    rng = np.random.default_rng(3)
    data = mujoco.MjData(model)
    ee = model.site("ee").id
    for _ in range(10):
        q = rng.normal(size=4)
        data.qpos[:] = 0
        data.qpos[0:3] = [0, 0, 3]
        data.qpos[3:7] = q / np.linalg.norm(q)
        data.qpos[qa] = [rng.uniform(lo + 0.05, hi - 0.05) for lo, hi in cfg.joint_ranges]
        data.qvel[:] = rng.normal(scale=[1, 1, 1, 2, 2, 2, 3, 3, 3])
        data.act[:4] = rng.uniform(0, cfg.thrust_max, 4)
        data.ctrl[:4] = data.act[:4]
        data.ctrl[4:] = data.qpos[qa]
        mujoco.mj_forward(model, data)
        F = rng.normal(scale=3.0, size=3)
        data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(model, data, F, np.zeros(3), data.site_xpos[ee], model.site_bodyid[ee], data.qfrc_applied)
        mujoco.mj_forward(model, data)
        x = state_from_mujoco(model, data, qa, qv)
        pdd, wd = wb.base_acc(x, data.act[:4], data.qacc[qv], F, np.zeros(3))
        assert np.allclose(np.asarray(pdd).ravel(), data.qacc[0:3], atol=1e-6)
        assert np.allclose(np.asarray(wd).ravel(), data.qacc[3:6], atol=1e-5)

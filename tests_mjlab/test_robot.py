import mujoco
import numpy as np

from ascento_mjlab.physics import PHYSICS_PROFILE
from ascento_mjlab.robot_cfg import DEFAULT_ASCENTO_CFG, JOINT_NAMES, ROBOT_XML, get_spec


def test_robot_asset_compiles_and_has_named_joints():
    assert ROBOT_XML.exists()
    model = get_spec().compile()
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    assert names == JOINT_NAMES


def test_robot_has_no_world_floor_or_legacy_mjx_name():
    spec = get_spec()
    assert spec.modelname == "ascento_guard2"
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1


def test_compiled_plant_enforces_profile_effort_on_all_actuated_joints():
    """The effective MuJoCo clamp, not only action scaling, is the authority."""
    model = DEFAULT_ASCENTO_CFG.build().spec.compile()
    expected = PHYSICS_PROFILE.peak_effort_nm

    assert model.nu == len(JOINT_NAMES)
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert joint_id >= 0
        assert model.jnt_actfrclimited[joint_id]
        assert np.allclose(model.jnt_actfrcrange[joint_id], (-expected, expected))


def test_high_motor_command_reaches_expected_generalized_joint_force():
    """Verify command, actuator output, and qfrc mapping on the compiled plant."""
    model = DEFAULT_ASCENTO_CFG.build().spec.compile()
    data = mujoco.MjData(model)
    data.ctrl[:] = PHYSICS_PROFILE.peak_effort_nm
    mujoco.mj_forward(model, data)

    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        dof_id = model.jnt_dofadr[joint_id]
        expected = data.actuator_force[actuator_id] * model.actuator_gear[actuator_id, 0]
        np.testing.assert_allclose(data.actuator_force[actuator_id], PHYSICS_PROFILE.peak_effort_nm)
        np.testing.assert_allclose(data.qfrc_actuator[dof_id], expected)

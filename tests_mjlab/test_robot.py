import mujoco

from ascento_mjlab.robot_cfg import JOINT_NAMES, ROBOT_XML, get_spec


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

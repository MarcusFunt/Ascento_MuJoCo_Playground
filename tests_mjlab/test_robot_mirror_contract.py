from pathlib import Path

import mujoco
import numpy as np

from ascento_mjlab.robot_cfg import ROBOT_XML


def _joint_address(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _supported_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:7] = (0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0)
    for name in ("left_hip", "left_knee", "right_hip", "right_knee"):
        qpos_address, _ = _joint_address(model, name)
        data.qpos[qpos_address] = -np.pi
    return data


def test_robot_xml_is_geometrically_mirrored_at_equal_leg_coordinates():
    assert Path(ROBOT_XML).is_file()
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = _supported_data(model)
    mujoco.mj_forward(model, data)

    for left_name, right_name in (
        ("left_thigh", "right_thigh"),
        ("left_shank", "right_shank"),
        ("left_wheel", "right_wheel"),
    ):
        left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_name)
        right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_name)
        left = data.xpos[left_id]
        right = data.xpos[right_id]
        assert np.allclose(left[[0, 2]], right[[0, 2]], atol=1.0e-7)
        assert np.isclose(left[1], -right[1], atol=1.0e-7)


def test_equal_leg_efforts_preserve_the_mirrored_generalized_direction():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = _supported_data(model)
    _, left_knee_dof = _joint_address(model, "left_knee")
    _, right_knee_dof = _joint_address(model, "right_knee")
    data.qfrc_applied[left_knee_dof] = 1.0
    data.qfrc_applied[right_knee_dof] = 1.0
    mujoco.mj_forward(model, data)

    _, left_hip_dof = _joint_address(model, "left_hip")
    _, right_hip_dof = _joint_address(model, "right_hip")
    assert np.isclose(data.qacc[left_hip_dof], data.qacc[right_hip_dof], atol=1.0e-7)
    assert np.isclose(data.qacc[left_knee_dof], data.qacc[right_knee_dof], atol=1.0e-7)

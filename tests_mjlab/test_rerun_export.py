"""Contract tests for Ascento reward capture and offline Rerun conversion."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from ascento_mjlab.robot_cfg import JOINT_NAMES, ROBOT_XML
from ascento_mjlab.tools.capture_motion import MotionRecorder
from ascento_mjlab.tools.export_rerun import validate_capture


def _episode_data():
    n = 3
    return {
        "time": np.arange(1, n + 1, dtype=np.float64) * 0.01,
        "root_pos": np.tile([0.0, 0.0, 0.75], (n, 1)),
        "root_quat": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "joint_pos": np.tile([-3.0, -3.0, 0.0, -3.0, -3.0, 0.0], (n, 1)),
        "root_lin_vel_w": np.tile([0.1, 0.0, 0.0], (n, 1)),
        "target_xy": np.tile([1.0, 0.0], (n, 1)),
        "speed_command": np.ones((n, 1)) * 0.2,
        "meta_joint_names": np.asarray(json.dumps(list(JOINT_NAMES))),
        "reward_terms_rate": np.tile([2.0, -0.5], (n, 1)),
        "reward_total": np.full(n, 0.015),
        "reward_dt": np.full(n, 0.01),
        "meta_reward_term_names": np.asarray(json.dumps(["progress", "penalty"])),
    }


def test_capture_records_authoritative_reward_and_navigation_state():
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_lin_vel_w=torch.tensor([[0.1, 0.0, 0.0]]),
            joint_pos=torch.zeros((1, 6)),
            joint_vel=torch.zeros((1, 6)),
            actuator_force=torch.zeros((1, 6)),
        )
    )

    class Commands:
        def get_command(self, name):
            if name == "speed":
                return torch.tensor([[0.3]])
            raise KeyError(name)

    reward_manager = SimpleNamespace(
        get_active_iterable_terms=lambda idx: [("progress", [2.0]), ("penalty", [-0.5])],
        _reward_buf=torch.tensor([0.015]),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        common_step_counter=1,
        step_dt=0.01,
        action_manager=SimpleNamespace(action=torch.zeros((1, 6))),
        command_manager=Commands(),
        reward_manager=reward_manager,
        ascento_world_target_state={
            "target_xy": torch.tensor([[1.0, 0.5]]),
            "target_yaw": torch.tensor([0.3]),
        },
    )
    recorder = SimpleNamespace(_env=env, frames=[], reward_term_names=("progress", "penalty"))
    MotionRecorder.record_post_step(recorder)
    frame = recorder.frames[0]
    np.testing.assert_allclose(frame["reward_terms_rate"], [2, -0.5])
    assert float(frame["reward_total"]) == pytest.approx(0.015)
    assert float(frame["reward_dt"]) == pytest.approx(0.01)
    np.testing.assert_allclose(frame["target_xy"], [1.0, 0.5])
    np.testing.assert_allclose(frame["root_lin_vel_w"], [0.1, 0, 0])
    np.testing.assert_allclose(frame["speed_command"], [0.3])


def test_validate_capture_checks_named_joint_mapping_and_reward_consistency():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    frames = validate_capture(_episode_data(), model)
    assert frames["count"] == 3
    for name, addr in zip(JOINT_NAMES, frames["joint_addrs"], strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert addr == model.jnt_qposadr[jid]
    assert frames["reward_term_names"] == ["progress", "penalty"]


def test_validate_capture_rejects_invalid_reward_accounting():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = _episode_data()
    data["reward_total"][1] += 0.05
    with pytest.raises(ValueError, match="Reward terms"):
        validate_capture(data, model)


def test_validate_capture_rejects_joint_name_mismatch():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = _episode_data()
    data["meta_joint_names"] = np.asarray(json.dumps(["nonsense"] * 6))
    with pytest.raises(ValueError, match="Joint names"):
        validate_capture(data, model)


def test_validate_capture_rejects_non_monotonic_time():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = _episode_data()
    data["time"][2] = data["time"][1]
    with pytest.raises(ValueError, match="non-monotonic"):
        validate_capture(data, model)


def test_real_rerun_export_if_dependencies_installed(tmp_path):
    pytest.importorskip("rerun")
    pytest.importorskip("rerun_loader_mjcf")
    from ascento_mjlab.tools.export_rerun import export_rerun

    source = tmp_path / "episode.npz"
    output = tmp_path / "episode.rrd"
    np.savez_compressed(source, **_episode_data())
    result = export_rerun(source, output)
    assert result["frames"] == 3
    assert result["reward_terms"] == 2
    assert result["target"] is True
    assert output.is_file()
    assert output.stat().st_size > 256

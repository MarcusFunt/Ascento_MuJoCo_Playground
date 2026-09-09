from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.metrics import (
    actuator_output_effort,
    commanded_effort,
    joint_applied_effort,
)


def test_effort_metrics_keep_command_output_and_joint_force_distinct():
    asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_effort_target=torch.tensor([[65.0, -30.0]]),
            actuator_force=torch.tensor([[40.0, -20.0]]),
            qfrc_actuator=torch.tensor([[38.0, -18.0]]),
        )
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0, 1], joint_ids=[0, 1])

    assert commanded_effort(env, asset_cfg).item() == pytest.approx(47.5)
    assert actuator_output_effort(env, asset_cfg).item() == pytest.approx(30.0)
    assert joint_applied_effort(env, asset_cfg).item() == pytest.approx(28.0)

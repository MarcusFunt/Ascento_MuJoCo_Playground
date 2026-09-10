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
        actuators=(
            SimpleNamespace(controller_requested_effort=torch.tensor([[65.0, -30.0, 10.0, -10.0]])),
            SimpleNamespace(controller_requested_effort=torch.tensor([[5.0, -5.0]])),
        ),
        data=SimpleNamespace(
            actuator_force=torch.tensor([[40.0, -20.0, 5.0, 10.0, -5.0, -10.0]]),
            qfrc_actuator=torch.tensor([[38.0, -18.0, 4.0, 8.0, -4.0, -8.0]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0, 1, 2, 3, 4, 5], joint_ids=[0, 1])

    assert commanded_effort(env, asset_cfg).item() == pytest.approx(20.833333)
    assert actuator_output_effort(env, asset_cfg).item() == pytest.approx(15.0)
    assert joint_applied_effort(env, asset_cfg).item() == pytest.approx(28.0)

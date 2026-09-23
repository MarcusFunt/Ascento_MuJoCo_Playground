from __future__ import annotations

import torch

from ascento_mjlab.structured_action import StructuredTargetAction


def test_structured_action_diagnostic_buffers_are_read_only_copies():
    action = StructuredTargetAction.__new__(StructuredTargetAction)
    action._raw_actions = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]])
    action._position_targets = torch.tensor([[0.01, 0.02, 0.03, 0.04]])
    action._velocity_targets = torch.tensor([[1.0, -2.0]])
    expected_raw = action._raw_actions.clone()
    expected_positions = action._position_targets.clone()
    expected_velocities = action._velocity_targets.clone()

    raw = action.raw_action
    positions = action.position_targets
    velocities = action.velocity_targets
    raw[0, 0] = 99.0
    positions[0, 0] = 99.0
    velocities[0, 0] = 99.0

    assert torch.equal(action._raw_actions, expected_raw)
    assert torch.equal(action._position_targets, expected_positions)
    assert torch.equal(action._velocity_targets, expected_velocities)

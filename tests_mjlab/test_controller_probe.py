import math

import torch

from ascento_mjlab.tools.controller_probe import (
    OPPOSING_TARGET_YAW_SIGN,
    _body_forward_xy,
    _step_count,
    _wrapped_difference,
    _yaw,
    characterize,
)


def test_controller_probe_uses_the_production_20_second_horizon():
    assert _step_count(20.0, 0.01) == 2_000


def test_controller_probe_body_frame_and_turn_convention_are_unambiguous():
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    clockwise = torch.tensor(
        [[math.cos(math.pi / 8.0), 0.0, 0.0, -math.sin(math.pi / 8.0)]]
    )

    assert torch.allclose(_body_forward_xy(identity), torch.tensor([[1.0, 0.0]]))
    yaw_change = _wrapped_difference(_yaw(clockwise), _yaw(identity))
    assert OPPOSING_TARGET_YAW_SIGN * yaw_change.item() > 0.0


def test_controller_probe_exercises_the_production_action_and_actuator_path():
    payload = characterize(device="cpu", duration_s=0.2, direction_duration_s=0.1)

    assert payload["passed"], payload
    assert payload["neutral"]["held"]
    assert payload["equal_positive"]["body_forward_velocity_m_s"] > 0.0
    assert payload["opposing_targets"]["yaw_rate_rad_s"] < 0.0
    assert payload["partial_wheel_pi_reset"]["passed"]

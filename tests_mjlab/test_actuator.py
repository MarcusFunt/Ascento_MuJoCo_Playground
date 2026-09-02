import torch

from ascento_mjlab.actuator import torque_speed_limit


def test_torque_speed_envelope_and_controller_guard():
  velocity = torch.tensor([[0.0, 6.0, 12.0, -11.0]])
  requested = torch.tensor([[40.0, 40.0, 40.0, -40.0]])
  result = torque_speed_limit(40.0, 12.0, 4.0, velocity, requested)

  assert torch.allclose(result[0, 0], torch.tensor(40.0))
  assert torch.allclose(result[0, 1], torch.tensor(0.0))
  assert torch.allclose(result[0, 2], torch.tensor(0.0))
  # Negative velocity and negative torque is motoring above the wheel/leg
  # controller limit, so it is guarded too.
  assert torch.allclose(result[0, 3], torch.tensor(0.0))


def test_braking_remains_available_above_controller_speed():
  velocity = torch.tensor([[6.0]])
  requested = torch.tensor([[-40.0]])
  result = torque_speed_limit(40.0, 12.0, 4.0, velocity, requested)
  assert result.item() < 0.0

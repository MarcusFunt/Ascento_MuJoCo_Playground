from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.commands import AscentoMotionCommandCfg
from ascento_mjlab.mdp.jump import (
  JumpSemantics,
  initialize_jump_state,
  update_jump_state,
)


def _jump_env(*, step_dt: float = 0.01):
  left = torch.ones((1, 1), dtype=torch.bool)
  right = torch.ones((1, 1), dtype=torch.bool)
  robot_data = SimpleNamespace(
    root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
    root_link_lin_vel_w=torch.zeros((1, 3)),
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=step_dt,
    scene={
      "left_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=left)),
      "right_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=right)),
      "robot": SimpleNamespace(data=robot_data),
    },
  )
  initialize_jump_state(env)
  return env, left, right, robot_data


def test_jump_semantics_keeps_terrain_behind_flat_ground_gate():
  semantics = JumpSemantics()
  assert semantics.takeoff_requires_both_wheels_airborne
  assert semantics.landing_is_first_subsequent_wheel_contact
  assert semantics.landing_impact_uses_precontact_vertical_speed
  assert semantics.terrain_enabled is False


def test_motion_command_has_one_shot_jump_configuration():
  cfg = AscentoMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    jump_probability=1.0,
  )
  assert cfg.jump_probability == 1.0


def test_motion_command_one_shot_is_explicitly_documented():
  cfg = AscentoMotionCommandCfg(
    entity_name="robot",
    resampling_time_range=(10.0, 10.0),
    jump_probability=1.0,
  )
  assert "one-step" in cfg.__doc__


def test_jump_state_defaults_to_active_environment_step_dt():
  env, left, right, _ = _jump_env(step_dt=0.037)
  left.zero_()
  right.zero_()

  update_jump_state(env)

  assert env.ascento_jump_state["air_time"].item() == pytest.approx(0.037)
  assert env.ascento_jump_state["takeoff"].item() == pytest.approx(1.0)


def test_landing_is_first_subsequent_contact_even_with_one_wheel():
  env, left, right, _ = _jump_env()
  left.zero_()
  right.zero_()
  update_jump_state(env)
  assert env.ascento_jump_state["airborne"].item()

  left.fill_(True)
  update_jump_state(env)

  assert env.ascento_jump_state["landing"].item() == pytest.approx(1.0)
  assert not env.ascento_jump_state["supported"].item()


def test_landing_impact_uses_last_airborne_velocity_not_post_contact_velocity():
  env, left, right, robot = _jump_env()
  left.zero_()
  right.zero_()
  robot.root_link_lin_vel_w[0, 2] = 1.2
  update_jump_state(env)

  robot.root_link_lin_vel_w[0, 2] = -3.4
  update_jump_state(env)
  assert env.ascento_jump_state["last_airborne_vz"].item() == pytest.approx(-3.4)

  left.fill_(True)
  robot.root_link_lin_vel_w[0, 2] = -0.2
  update_jump_state(env)

  assert env.ascento_jump_state["landing"].item() == pytest.approx(1.0)
  assert env.ascento_jump_state["landing_preimpact_vz"].item() == pytest.approx(-3.4)

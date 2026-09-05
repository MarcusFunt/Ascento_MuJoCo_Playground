from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.commands import AscentoMotionCommandCfg
from ascento_mjlab.mdp.jump import (
  JumpSemantics,
  PHASE_CROUCH,
  PHASE_FLIGHT,
  initialize_jump_state,
  update_jump_state,
)


class _MotionTerm:
  def __init__(self):
    self.command = torch.zeros((1, 6))
    self.jump_generation = torch.zeros(1, dtype=torch.long)


class _CommandManager:
  def __init__(self, term):
    self.term = term

  def get_term(self, name):
    if name != "motion":
      raise KeyError(name)
    return self.term


def _jump_env(*, step_dt: float = 0.01):
  left = torch.ones((1, 1), dtype=torch.bool)
  right = torch.ones((1, 1), dtype=torch.bool)
  robot_data = SimpleNamespace(
    root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
    root_link_lin_vel_w=torch.zeros((1, 3)),
    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    body_link_pos_w=torch.tensor([[[0.0, 0.1, 0.25], [0.0, -0.1, 0.25]]]),
  )
  robot = SimpleNamespace(
    data=robot_data,
    body_names=("left_wheel", "right_wheel"),
  )
  motion = _MotionTerm()
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=step_dt,
    command_manager=_CommandManager(motion),
    scene={
      "left_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=left)),
      "right_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=right)),
      "robot": robot,
    },
  )
  initialize_jump_state(env)
  return env, left, right, robot_data, motion


def _request_jump(motion, *, distance: float = 0.2):
  motion.command[0] = torch.tensor([0.0, 0.0, 0.75, 1.0, 0.2, distance])
  motion.jump_generation[0] += 1


def test_jump_semantics_keeps_terrain_behind_flat_ground_gate():
  semantics = JumpSemantics()
  assert semantics.takeoff_requires_both_wheels_airborne
  assert semantics.landing_is_first_subsequent_wheel_contact
  assert semantics.landing_impact_uses_precontact_vertical_speed
  assert semantics.jump_distance_is_takeoff_heading_relative
  assert semantics.clearance_uses_simultaneous_limiting_wheel
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
  env, left, right, _, motion = _jump_env(step_dt=0.037)
  _request_jump(motion)
  update_jump_state(env)
  assert env.ascento_jump_state["phase"].item() == PHASE_CROUCH

  left.zero_()
  right.zero_()
  update_jump_state(env)

  assert env.ascento_jump_state["air_time"].item() == pytest.approx(0.037)
  assert env.ascento_jump_state["takeoff"].item() == pytest.approx(1.0)
  assert env.ascento_jump_state["phase"].item() == PHASE_FLIGHT


def test_unrequested_contact_loss_is_not_a_takeoff_event():
  env, left, right, _, _ = _jump_env()
  left.zero_()
  right.zero_()
  update_jump_state(env)

  assert env.ascento_jump_state["airborne"].item()
  assert env.ascento_jump_state["takeoff"].item() == 0.0


def test_landing_is_first_subsequent_contact_even_with_one_wheel():
  env, left, right, _, motion = _jump_env()
  _request_jump(motion)
  update_jump_state(env)
  left.zero_()
  right.zero_()
  update_jump_state(env)
  assert env.ascento_jump_state["airborne"].item()

  left.fill_(True)
  update_jump_state(env)

  assert env.ascento_jump_state["landing"].item() == pytest.approx(1.0)
  assert not env.ascento_jump_state["supported"].item()


def test_landing_impact_uses_last_airborne_velocity_not_post_contact_velocity():
  env, left, right, robot, motion = _jump_env()
  _request_jump(motion)
  update_jump_state(env)
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


def test_distance_is_measured_in_takeoff_heading_frame():
  env, left, right, robot, motion = _jump_env()
  _request_jump(motion, distance=0.30)
  update_jump_state(env)
  left.zero_()
  right.zero_()
  update_jump_state(env)

  robot.root_link_pos_w[0, 0] = 0.24
  update_jump_state(env)
  left.fill_(True)
  update_jump_state(env)

  assert env.ascento_jump_state["landing_distance_error"].item() == pytest.approx(-0.06)


def test_clearance_uses_lower_wheel_at_each_airborne_sample():
  env, left, right, robot, motion = _jump_env()
  _request_jump(motion)
  update_jump_state(env)
  left.zero_()
  right.zero_()
  robot.body_link_pos_w[0, 0, 2] = 0.45
  robot.body_link_pos_w[0, 1, 2] = 0.30
  update_jump_state(env)

  assert env.ascento_jump_state["limiting_wheel_clearance"].item() == pytest.approx(0.05)

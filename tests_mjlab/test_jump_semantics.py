from types import SimpleNamespace

import pytest
import torch

from ascento_mjlab.mdp.commands import AscentoMotionCommandCfg
from ascento_mjlab.mdp.jump import JumpSemantics, update_jump_state


def test_jump_semantics_keeps_terrain_behind_flat_ground_gate():
  semantics = JumpSemantics()
  assert semantics.takeoff_requires_both_wheels_airborne
  assert semantics.landing_is_first_subsequent_wheel_contact
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
  found = torch.zeros((1, 1), dtype=torch.bool)
  state = {
    "supported": torch.ones(1, dtype=torch.bool),
    "airborne": torch.zeros(1, dtype=torch.bool),
    "takeoff": torch.zeros(1),
    "landing": torch.zeros(1),
    "air_time": torch.zeros(1),
    "takeoff_height": torch.zeros(1),
  }
  env = SimpleNamespace(
    step_dt=0.037,
    ascento_jump_state=state,
    scene={
      "left_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=found)),
      "right_wheel_contact": SimpleNamespace(data=SimpleNamespace(found=found)),
      "robot": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]))
      ),
    },
  )

  update_jump_state(env, None)

  assert state["air_time"].item() == pytest.approx(0.037)

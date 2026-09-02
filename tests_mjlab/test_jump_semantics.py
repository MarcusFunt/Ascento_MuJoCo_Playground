from ascento_mjlab.mdp.commands import AscentoMotionCommandCfg
from ascento_mjlab.mdp.jump import JumpSemantics


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

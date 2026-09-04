import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401


def test_balance_env_is_six_effort_flat_ground():
  cfg = load_env_cfg("Ascento-Balance-Flat")
  assert cfg.decimation == 5
  assert cfg.sim.mujoco.timestep == 0.002
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, _ = env.reset()
  assert obs["actor"].shape[-1] == 38
  assert obs["critic"].shape[-1] == 47
  obs, reward, terminated, truncated, _ = env.step(
    torch.zeros((cfg.scene.num_envs, 6))
  )
  assert torch.isfinite(obs["actor"]).all()
  assert torch.isfinite(reward).all()
  assert not terminated.any()
  assert terminated.shape == truncated.shape == (cfg.scene.num_envs,)
  env.close()


def test_balance_action_contract_reaches_40_nm_and_penalizes_drift():
  cfg = load_env_cfg("Ascento-Balance-Flat")
  cfg.scene.num_envs = 1

  action_cfg = cfg.actions["effort"]
  assert action_cfg.scale == 40.0
  assert action_cfg.clip == {".*": (-40.0, 40.0)}
  assert cfg.rewards["planar_speed"].weight == pytest.approx(-0.2)

  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.action_manager.process_action(torch.ones((1, 6)))
  action_term = env.action_manager.get_term("effort")
  assert torch.allclose(action_term._processed_actions, torch.full((1, 6), 40.0))
  env.close()


def test_balance_rl_config_enforces_normalized_actions_and_instrumented_ppo():
  cfg = load_rl_cfg("Ascento-Balance-Flat")

  assert cfg.clip_actions == 1.0
  assert cfg.algorithm.class_name == "ascento_mjlab.ppo:InstrumentedPPO"


def test_velocity_stage_has_no_reward_that_penalizes_its_commands():
  cfg = load_env_cfg("Ascento-Velocity-Flat")

  assert set(cfg.commands) == {"twist", "height"}
  assert "height" not in cfg.rewards
  assert "planar_speed" not in cfg.rewards
  assert "track_velocity" in cfg.rewards
  assert "track_height" in cfg.rewards
  assert "twist_command" in cfg.observations["actor"].terms
  assert "height_command" in cfg.observations["actor"].terms


def test_recovery_stage_exports_executable_success_metric_and_training_pushes():
  cfg = load_env_cfg("Ascento-Recovery-Flat")
  play_cfg = load_env_cfg("Ascento-Recovery-Flat", play=True)

  assert "recovery_success" in cfg.metrics
  assert "recovery_push" in cfg.events
  assert cfg.events["recovery_push"].mode == "interval"
  assert "recovery_push" not in play_cfg.events


@pytest.mark.parametrize(
  "task_id",
  [
    "Ascento-Balance-Flat",
    "Ascento-Velocity-Flat",
    "Ascento-Recovery-Flat",
    "Ascento-Jump-Flat",
  ],
)
def test_all_flat_task_configs_construct_and_step(task_id):
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  obs, _ = env.reset()
  obs, reward, terminated, truncated, _ = env.step(torch.zeros((1, 6)))
  assert torch.isfinite(obs["actor"]).all()
  assert torch.isfinite(reward).all()
  assert terminated.shape == truncated.shape == (1,)
  env.close()


def test_jump_state_is_synchronized_before_jump_dependent_rewards():
  cfg = load_env_cfg("Ascento-Jump-Flat")

  assert "update_jump_state" not in cfg.events
  reward_names = list(cfg.rewards)
  assert reward_names.index("jump_state_sync") < reward_names.index("takeoff")
  assert reward_names.index("jump_state_sync") < reward_names.index("landing")
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.01)


def test_custom_sim_timestep_reaches_actuator():
  cfg = load_env_cfg("Ascento-Balance-Flat")
  cfg.scene.num_envs = 1
  cfg.sim.mujoco.timestep = 0.007

  env = ManagerBasedRlEnv(cfg, device="cpu")

  assert {act._physics_dt for act in env.scene["robot"].actuators} == {0.007}
  env.close()

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

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
  obs, reward, terminated, truncated, _ = env.step(torch.zeros((cfg.scene.num_envs, 6)))
  assert torch.isfinite(obs["actor"]).all()
  assert torch.isfinite(reward).all()
  assert not terminated.any()
  assert terminated.shape == truncated.shape == (cfg.scene.num_envs,)


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

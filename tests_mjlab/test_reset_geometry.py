import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.mdp.events import flat_ground_wheel_bottom_heights


def test_recovery_resets_anchor_lowest_wheel_to_flat_support_plane():
  cfg = load_env_cfg("Ascento-Recovery-Flat")
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observed = []
    for _ in range(16):
      env.reset()
      bottoms = flat_ground_wheel_bottom_heights(env)
      observed.append(bottoms.amin(dim=1))
    lowest = torch.cat(observed)

    # The support-aware reset should eliminate both penetration and whole-robot
    # hovering caused solely by orientation perturbation at a fixed root height.
    assert torch.max(torch.abs(lowest)).item() <= 2.0e-3
  finally:
    env.close()


def test_all_flat_tasks_use_support_aware_root_resets():
  for task in (
    "Ascento-Balance-Flat",
    "Ascento-Velocity-Flat",
    "Ascento-Recovery-Flat",
    "Ascento-Jump-Flat",
  ):
    cfg = load_env_cfg(task)
    assert cfg.events["reset_supported_pose"].func.__name__ == "reset_root_state_supported"

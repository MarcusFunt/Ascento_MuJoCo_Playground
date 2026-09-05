"""Print pinned runtime versions and exercise one Warp rollout."""

from __future__ import annotations

from importlib.metadata import version

import mujoco
import torch
import warp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import ascento_mjlab.tasks  # noqa: F401


def main() -> None:
    print(f"torch={torch.__version__}")
    print(
        f"cuda={torch.version.cuda} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"gpu={torch.cuda.get_device_name(0)} vram_gib={properties.total_memory / 2**30:.1f}")
    print(f"mujoco={mujoco.__version__} mujoco_warp={version('mujoco-warp')}")
    print(f"warp={warp.config.version} mjlab={version('mjlab')} rsl_rl={version('rsl-rl-lib')}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_env_cfg("Ascento-Balance-Flat")
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device=device)
    observations, _ = env.reset()
    for _ in range(100):
        observations, reward, _, _, _ = env.step(torch.zeros((1, 6), device=device))
    assert all(torch.isfinite(value).all() for value in observations.values())
    assert torch.isfinite(reward).all()
    print(f"smoke=ok actor_obs={tuple(observations['actor'].shape)} reward={reward.item():.5f}")


if __name__ == "__main__":
    main()

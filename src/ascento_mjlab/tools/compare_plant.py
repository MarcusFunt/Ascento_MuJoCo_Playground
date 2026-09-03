"""Migration-only ordinary MuJoCo versus Warp plant comparison.

This diagnostic is deliberately not part of CI or the permanent runtime path.
Archive it after Gates D/E unless it catches a specific plant regression.
"""

from __future__ import annotations

import argparse

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import ascento_mjlab.tasks  # noqa: F401


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=100)
  args = parser.parse_args()
  cfg = load_env_cfg("Ascento-Balance-Flat", play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset(seed=0)
  ordinary = mujoco.MjData(env.sim.mj_model)
  ordinary.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  ordinary.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
  ordinary.ctrl[:] = 0.0
  mujoco.mj_forward(env.sim.mj_model, ordinary)
  for _ in range(args.steps):
    for _ in range(cfg.decimation):
      mujoco.mj_step(env.sim.mj_model, ordinary)
    env.step(torch.zeros((1, 6)))
  warp_pos = env.scene["robot"].data.root_link_pos_w[0].cpu()
  ordinary_pos = torch.from_numpy(ordinary.qpos[:3])
  print("ordinary root position:", ordinary_pos.tolist())
  print("warp root position:", warp_pos.tolist())
  print("delta_m:", torch.linalg.vector_norm(ordinary_pos - warp_pos).item())
  print("diagnostic_only=true")
  env.close()


if __name__ == "__main__":
  main()

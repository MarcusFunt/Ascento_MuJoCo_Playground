"""Inspect the compiled Ascento scene and actuator mapping."""

from __future__ import annotations

import argparse

import mujoco
from mjlab.scene import Scene

from ascento_mjlab.robot_cfg import SIM_CFG
from ascento_mjlab.tasks.balance.env_cfg import _scene


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=1)
  args = parser.parse_args()
  scene = Scene(_scene(args.num_envs), device="cpu")
  model = scene.compile()
  print(f"MuJoCo {mujoco.__version__}")
  print(f"mjlab scene: {args.num_envs} env(s), nq={model.nq}, nv={model.nv}, nu={model.nu}, ngeom={model.ngeom}")
  print("actuators:", [model.actuator(i).name for i in range(model.nu)])
  print("simulation timestep:", SIM_CFG.mujoco.timestep)


if __name__ == "__main__":
  main()

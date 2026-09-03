"""Capture multiple flat-ground motion takes through mjlab's RecorderManager."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

import ascento_mjlab.tasks  # noqa: F401


class MotionRecorder(RecorderTerm):
  """Collect named state channels for one environment and save them as NPZ."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.frames = []

  def record_post_step(self) -> None:
    env = self._env
    robot = env.scene["robot"]
    self.frames.append(
      {
        "time": env.common_step_counter * env.step_dt,
        "root_pos": robot.data.root_link_pos_w[0].detach().cpu().numpy().copy(),
        "root_quat": robot.data.root_link_quat_w[0].detach().cpu().numpy().copy(),
        "joint_pos": robot.data.joint_pos[0].detach().cpu().numpy().copy(),
        "joint_vel": robot.data.joint_vel[0].detach().cpu().numpy().copy(),
        "effort": robot.data.actuator_force[0].detach().cpu().numpy().copy(),
        "action": env.action_manager.action[0].detach().cpu().numpy().copy(),
      }
    )

  def export(self, path: Path, *, metadata: dict[str, str]) -> None:
    if not self.frames:
      raise RuntimeError("No frames were captured")
    keys = self.frames[0].keys()
    arrays = {key: np.stack([frame[key] for frame in self.frames]) for key in keys}
    arrays.update({f"meta_{key}": np.asarray(value) for key, value in metadata.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Ascento-Balance-Flat")
  parser.add_argument("--checkpoint", type=Path, default=None, help="RSL-RL checkpoint to play")
  parser.add_argument("--takes", type=int, default=1)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--output-dir", type=Path, default=Path("captures"))
  parser.add_argument("--video-dir", type=Path, default=None, help="Optional MP4 output directory")
  parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()
  if args.takes < 1 or args.steps < 1:
    parser.error("--takes and --steps must be positive")
  if args.checkpoint is not None and not args.checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {args.checkpoint}")

  for take in range(args.takes):
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = take
    cfg.recorders = {"motion": RecorderTermCfg(func=MotionRecorder, params={})}
    base_env = ManagerBasedRlEnv(
      cfg,
      device=args.device,
      render_mode="rgb_array" if args.video_dir is not None else None,
    )
    raw_env = (
      VideoRecorder(
        base_env,
        video_folder=args.video_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix=f"take-{take:03d}",
        disable_logger=True,
      )
      if args.video_dir is not None
      else base_env
    )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=load_rl_cfg(args.task).clip_actions)
    if args.checkpoint is None:
      def policy(obs):
        del obs
        return torch.zeros((1, 6), device=args.device)
    else:
      agent_cfg = load_rl_cfg(args.task)
      runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
      runner = runner_cls(env, asdict(agent_cfg), device=args.device)
      runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
      )
      policy = runner.get_inference_policy(device=args.device)
    obs, _ = env.reset()
    for _ in range(args.steps):
      obs, _, _, _ = env.step(policy(obs))
    recorder = raw_env.recorder_manager.get_term("motion")
    recorder.export(
      args.output_dir / f"take_{take:03d}.npz",
      metadata={"task": args.task, "seed": str(take), "fps": str(round(1.0 / raw_env.step_dt))},
    )
    env.close()


if __name__ == "__main__":
  main()

"""Capture multiple flat-ground motion takes through mjlab's RecorderManager."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.physics import PHYSICS_PROFILE


def _jump_state_array(state: dict[str, torch.Tensor]) -> np.ndarray:
  """Copy the four public jump-state scalars to host memory as float32."""
  values = torch.stack(
    [
      state["airborne"][0],
      state["takeoff"][0],
      state["landing"][0],
      state["air_time"][0],
    ]
  )
  return values.detach().to(device="cpu", dtype=torch.float32).numpy().copy()


def _configure_capture_cfg(cfg: Any, *, take: int) -> Any:
  """Configure a task for one continuous, non-auto-reset capture take."""
  cfg.scene.num_envs = 1
  cfg.seed = take
  cfg.auto_reset = False
  cfg.recorders = {"motion": RecorderTermCfg(func=MotionRecorder, params={})}
  return cfg


def _run_capture_steps(
  env: Any,
  policy: Callable[[Any], torch.Tensor],
  *,
  steps: int,
) -> tuple[int, bool]:
  """Run until the requested length or the first terminal/truncated step."""
  obs, _ = env.reset()
  for step in range(1, steps + 1):
    obs, _, dones, _ = env.step(policy(obs))
    if bool(torch.any(dones).item()):
      return step, True
  return steps, False


class MotionRecorder(RecorderTerm):
  """Collect named state channels for one environment and save them as NPZ."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.frames = []

  def record_post_step(self) -> None:
    env = self._env
    robot = env.scene["robot"]
    frame = {
      "time": env.common_step_counter * env.step_dt,
      "root_pos": robot.data.root_link_pos_w[0].detach().cpu().numpy().copy(),
      "root_quat": robot.data.root_link_quat_w[0].detach().cpu().numpy().copy(),
      "joint_pos": robot.data.joint_pos[0].detach().cpu().numpy().copy(),
      "joint_vel": robot.data.joint_vel[0].detach().cpu().numpy().copy(),
      "effort": robot.data.actuator_force[0].detach().cpu().numpy().copy(),
      "action": env.action_manager.action[0].detach().cpu().numpy().copy(),
    }
    contacts = []
    for sensor_name in ("left_wheel_contact", "right_wheel_contact"):
      try:
        sensor = env.scene[sensor_name]
      except KeyError:
        sensor = None
      if sensor is not None and sensor.data.found is not None:
        contacts.append(float(sensor.data.found[0].flatten().any()))
    if len(contacts) == 2:
      frame["contacts"] = np.asarray(contacts, dtype=np.float32)
    if hasattr(env, "ascento_jump_state"):
      frame["jump_state"] = _jump_state_array(env.ascento_jump_state)
      if "landing_preimpact_vz" in env.ascento_jump_state:
        frame["landing_preimpact_vz"] = (
          env.ascento_jump_state["landing_preimpact_vz"][0]
          .detach()
          .cpu()
          .numpy()
          .copy()
        )
    for command_name in ("motion", "twist"):
      try:
        command = env.command_manager.get_command(command_name)
      except (AttributeError, KeyError):
        command = None
      if command is not None:
        frame["command"] = command[0].detach().cpu().numpy().copy()
        break
    try:
      height_command = env.command_manager.get_command("height")
    except (AttributeError, KeyError):
      height_command = None
    if height_command is not None:
      frame["height_command"] = height_command[0].detach().cpu().numpy().copy()
    self.frames.append(frame)

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
  parser.add_argument(
    "--checkpoint", type=Path, default=None, help="RSL-RL checkpoint to play"
  )
  parser.add_argument("--takes", type=int, default=1)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--output-dir", type=Path, default=Path("captures"))
  parser.add_argument(
    "--video-dir", type=Path, default=None, help="Optional MP4 output directory"
  )
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()
  if args.takes < 1 or args.steps < 1:
    parser.error("--takes and --steps must be positive")
  if args.checkpoint is not None and not args.checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {args.checkpoint}")

  checkpoint_hash = ""
  if args.checkpoint is not None:
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

  for take in range(args.takes):
    cfg = _configure_capture_cfg(load_env_cfg(args.task, play=True), take=take)
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
    captured_steps, ended_on_done = _run_capture_steps(env, policy, steps=args.steps)
    recorder = base_env.recorder_manager.get_term("motion")
    recorder.export(
      args.output_dir / f"take_{take:03d}.npz",
      metadata={
        "task": args.task,
        "seed": str(take),
        "fps": str(round(1.0 / base_env.step_dt)),
        "physics_timestep_s": str(float(base_env.cfg.sim.mujoco.timestep)),
        "policy_timestep_s": str(float(base_env.step_dt)),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else "",
        "model_sha256": checkpoint_hash,
        "physics_profile": PHYSICS_PROFILE.name,
        "captured_steps": str(captured_steps),
        "ended_on_done": str(ended_on_done).lower(),
      },
    )
    env.close()


if __name__ == "__main__":
  main()

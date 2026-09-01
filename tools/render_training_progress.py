"""Periodically render deterministic rollouts from live Brax PPO checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ASCENTO_JAX_PLATFORM", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import imageio.v2 as imageio
import mujoco
import numpy as np
from brax.training.agents.ppo import checkpoint, networks
from brax.training.acme import running_statistics

from training.bounded_distribution import BoundedNormalTanhDistribution

from training.ppo_config import build_environment


def load_checkpoint_policy(checkpoint_path: Path, env, hidden_sizes: tuple[int, ...]):
    """Builds the exact deterministic policy architecture used by training."""
    params = checkpoint.load(str(checkpoint_path.resolve()))
    net = networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        policy_obs_key="state",
        value_obs_key="state",
    ).replace(parametric_action_distribution=BoundedNormalTanhDistribution(env.action_size))
    return networks.make_inference_fn(net)(params, deterministic=True)


def numeric_checkpoints(checkpoint_dir: Path):
    return sorted(
        (path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ) if checkpoint_dir.exists() else []


def render_preview(env, policy, seed: int, output: Path, steps: int):
    """Writes a tiled PNG preview without spawning an ffmpeg subprocess."""
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    action_fn = jax.jit(lambda obs: policy(obs[None, :], jax.random.PRNGKey(0))[0][0])
    state = reset(jax.random.PRNGKey(seed))
    renderer = mujoco.Renderer(env.mj_model, height=240, width=426)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.mj_model, camera)
    camera.azimuth, camera.elevation, camera.distance = 135, -15, 1.65
    data, frames, returns, done_steps = mujoco.MjData(env.mj_model), [], 0.0, 0
    heights, gravity_z, max_abs_qvel, action_saturation = [], [], [], []
    sample_every = max(1, steps // 12)
    try:
        for index in range(steps):
            state = step(state, action_fn(state.obs))
            returns += float(state.reward)
            done_steps += int(state.done)
            heights.append(float(state.metrics["metric/body_height"]))
            gravity_z.append(float(state.metrics["metric/gravity_z"]))
            max_abs_qvel.append(float(state.metrics["metric/max_abs_qvel"]))
            action_saturation.append(float(np.mean(np.abs(np.asarray(action)) >= 0.98)))
            if index % sample_every == 0:
                data.qpos[:] = np.asarray(jax.device_get(state.data.qpos))
                data.qvel[:] = np.asarray(jax.device_get(state.data.qvel))
                mujoco.mj_forward(env.mj_model, data)
                camera.lookat[:] = data.qpos[:3]
                camera.lookat[2] += 0.12
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())
    finally:
        renderer.close()
    rows = [np.concatenate(frames[index:index + 4], axis=1) for index in range(0, len(frames), 4)]
    preview = np.concatenate(rows, axis=0)
    imageio.imwrite(output, preview)
    return {
        "seed": seed,
        "path": str(output.resolve()),
        "return": returns,
        "done_steps": done_steps,
        "mean_height": float(np.mean(heights)),
        "min_height": float(np.min(heights)),
        "mean_gravity_z": float(np.mean(gravity_z)),
        "min_gravity_z": float(np.min(gravity_z)),
        "max_abs_qvel": float(np.max(max_abs_qvel)),
        "action_saturation_fraction": float(np.mean(action_saturation)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="balance")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--hidden-sizes", default="128,128")
    args = parser.parse_args()
    hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(",") if value)
    env, _ = build_environment(args.stage, episode_length=args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    manifest_path = args.output_dir / "progress_renders.jsonl"
    while True:
        for checkpoint_path in numeric_checkpoints(args.checkpoint_dir):
            if checkpoint_path.name in seen:
                continue
            try:
                policy = load_checkpoint_policy(checkpoint_path, env, hidden_sizes)
                output = args.output_dir / f"{args.stage}_step_{int(checkpoint_path.name):09d}.png"
                report = render_preview(env, policy, args.seed, output, args.steps)
                record = {"checkpoint": str(checkpoint_path.resolve()), "step": int(checkpoint_path.name), **report}
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print("PROGRESS_RENDER " + json.dumps(record), flush=True)
                seen.add(checkpoint_path.name)
            except Exception as error:  # checkpoint can still be asynchronously finalized
                print(f"PROGRESS_RENDER_RETRY step={checkpoint_path.name} error={error}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

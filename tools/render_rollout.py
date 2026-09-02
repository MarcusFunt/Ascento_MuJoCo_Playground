"""Render deterministic MJX policy rollouts to MP4 without Blender."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ASCENTO_JAX_PLATFORM", "cuda")
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import imageio.v2 as imageio
import jax
import mujoco
import numpy as np

from evaluation.evaluate import load_policy
from training.ppo_config import build_environment


def render_seed(env, policy, seed: int, output: Path, steps: int, fps: int):
    """Replays one deterministic policy trajectory through MuJoCo's renderer."""
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    action_fn = jax.jit(lambda obs: policy(obs[None, :], jax.random.PRNGKey(0))[0][0])
    state = reset(jax.random.PRNGKey(seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.mj_model, height=360, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.mj_model, camera)
    camera.azimuth, camera.elevation, camera.distance = 135, -15, 1.65
    data = mujoco.MjData(env.mj_model)
    returns, dones = 0.0, 0
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for _ in range(steps):
            action = action_fn(state.obs)
            state = step(state, action)
            returns += float(state.reward)
            dones += int(state.done)
            data.qpos[:] = np.asarray(jax.device_get(state.data.qpos))
            data.qvel[:] = np.asarray(jax.device_get(state.data.qvel))
            mujoco.mj_forward(env.mj_model, data)
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] += 0.12
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    return {"seed": seed, "path": str(output.resolve()), "return": returns, "done_steps": dones}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="balance")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=100)
    args = parser.parse_args()
    env, _ = build_environment(args.stage, episode_length=args.steps)
    policy, manifest = load_policy(args.artifact, env.observation_size, env.action_size)
    reports = []
    for seed in (int(value) for value in args.seeds.split(",") if value.strip()):
        reports.append(render_seed(env, policy, seed, args.output_dir / f"{args.stage}_seed_{seed}.mp4", args.steps, args.fps))
    (args.output_dir / "render_manifest.json").write_text(
        json.dumps({"stage": args.stage, "artifact": str(args.artifact.resolve()), "policy": manifest, "videos": reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

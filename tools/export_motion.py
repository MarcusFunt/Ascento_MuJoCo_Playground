"""Exports deterministic policy motion to NPZ plus replay metadata; no renderer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("ASCENTO_JAX_PLATFORM", "cuda")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jp
import numpy as np

from evaluation.evaluate import load_policy
from training.ppo_config import build_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="balance")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="output .npz path")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()
    env, _ = build_environment(args.stage, episode_length=args.steps)
    policy, manifest = load_policy(args.artifact, env.observation_size, env.action_size)
    state = env.reset(jax.random.PRNGKey(args.seed))

    def rollout(current, _):
        action, _ = policy(current.obs[None, :], jax.random.PRNGKey(0))
        nxt = env.step(current, action[0])
        sample = (
            nxt.data.time,
            nxt.data.qpos[:7],
            nxt.data.qvel[:6],
            nxt.data.qpos[7:13],
            nxt.data.qvel[6:12],
            action[0],
            nxt.info["last_torque"],
            nxt.info["wheel_contact"].astype(jp.float32),
            nxt.info["wheel_contact_force"],
            nxt.info["jump_phase"],
            nxt.info["jump_takeoff_event"],
            nxt.info["jump_landing_event"],
            nxt.info["jump_success_event"],
            nxt.info["jump_air_time"],
            nxt.info["jump_target_x"],
            nxt.info["command"],
        )
        return nxt, sample

    _, trace = jax.jit(lambda initial: jax.lax.scan(rollout, initial, None, length=args.steps))(state)
    values = [np.asarray(value) for value in jax.device_get(trace)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time=values[0], root_qpos=values[1], root_qvel=values[2], joint_qpos=values[3],
        joint_qvel=values[4], actions=values[5], applied_torque=values[6],
        wheel_contact=values[7], wheel_contact_force=values[8], jump_phase=values[9],
        jump_takeoff_event=values[10], jump_landing_event=values[11],
        jump_success_event=values[12], jump_air_time=values[13], jump_target_x=values[14],
        command=values[15],
    )
    metadata = {
        "checkpoint": str(args.artifact.resolve()),
        "stage": args.stage,
        "seed": args.seed,
        "physics_timestep": 0.002,
        "policy_timestep": 0.01,
        "model": env.xml_path,
        "manifest": manifest,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

"""Deterministic, seed-fixed evaluation for every direct-torque stage."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path


def configure_backend():
    requested = os.environ.get("ASCENTO_JAX_PLATFORM", "cuda").lower()
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if requested == "cuda" else "cpu")


configure_backend()
import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks

from training.ppo_config import build_environment


def load_policy(artifact: Path, observation_size: int, action_size: int):
    with (artifact / "policy_params.pkl").open("rb") as handle:
        params = pickle.load(handle)
    manifest = json.loads((artifact / "training_manifest.json").read_text())
    if manifest.get("observation_size", observation_size) != observation_size:
        raise ValueError("artifact observation schema does not match the fixed 49-value curriculum schema")
    hidden = tuple(manifest["network_hidden_layer_sizes"])
    net = networks.make_ppo_networks(
        observation_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=hidden,
        policy_obs_key="state",
        value_obs_key="state",
    )
    return networks.make_inference_fn(net)(params, deterministic=True), manifest


def evaluate(stage: str, artifact: Path, episodes: int = 50, steps: int = 600, seed: int = 91):
    env, _ = build_environment(stage, episode_length=steps)
    policy, manifest = load_policy(artifact, env.observation_size, env.action_size)
    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    state = reset(keys)

    def rollout(current, _):
        action, _ = policy(current.obs, jax.random.PRNGKey(0))
        nxt = step(current, action)
        return nxt, (
            nxt.reward,
            nxt.done,
            nxt.obs[:, :3],
            nxt.data.qpos[:, 2],
            action,
            nxt.info["jump_takeoff_event"],
            nxt.info["jump_success_event"],
            nxt.info["jump_wheel_apex_height"],
            nxt.info["jump_landing_vz"],
        )

    _, trace = jax.jit(lambda initial: jax.lax.scan(rollout, initial, None, length=steps))(state)
    reward, done, gravity, height, action, takeoff, success, wheel_apex, landing_vz = map(np.asarray, jax.device_get(trace))
    survival = (1.0 - done).cumprod(axis=0).sum(axis=0)
    tilt = np.arccos(np.clip(-gravity[..., 2], -1.0, 1.0))
    action_jitter = np.diff(action, axis=0)
    takeoff_rate = float((takeoff.sum(axis=0) > 0).mean())
    recovery_success = float((success.sum(axis=0) > 0).mean())
    success_mask = success > 0
    success_steps = np.where(success_mask.any(axis=0), success_mask.argmax(axis=0) + 1, np.nan)
    result = {
        "stage": stage,
        "artifact": str(artifact.resolve()),
        "seed": seed,
        "episodes": episodes,
        "steps": steps,
        "metrics": {
            "survival_rate": float((survival >= steps).mean()),
            "mean_survival_steps": float(survival.mean()),
            "mean_return": float(reward.sum(axis=0).mean()),
            "rms_tilt": float(np.sqrt(np.mean(np.square(tilt)))),
            "height_error": float(np.mean(np.abs(height - 0.75))),
            "action_jitter": float(np.sqrt(np.mean(np.square(action_jitter)))) if len(action_jitter) else 0.0,
            "takeoff_rate": takeoff_rate,
            "wheel_clearance": float(np.maximum(0.0, wheel_apex.max(axis=0) - 0.25).mean()),
            "landing_speed": float(np.abs(landing_vz[landing_vz != 0]).mean()) if np.any(landing_vz != 0) else float("inf"),
            "recovery_success_rate": recovery_success,
            "failure_rate": float(1.0 - (survival >= steps).mean()),
            "median_recovery_time": float(np.nanmedian(success_steps) * 0.01) if np.any(success_mask) else float("inf"),
        },
        "manifest_stage": manifest.get("stage"),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="balance")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.stage, args.artifact, args.episodes, args.steps, args.seed)
    output = args.output or args.artifact / "evaluation.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

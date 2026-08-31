"""CUDA-first Brax PPO training entry point for nominal upright balance."""
import argparse
import json
import os
from pathlib import Path

def configure_backend():
    requested = os.environ.get("ASCENTO_JAX_PLATFORM", "cuda").lower()
    if requested == "cuda":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
    elif requested == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    else:
        raise ValueError("ASCENTO_JAX_PLATFORM must be cuda or cpu")

configure_backend()
import jax
import jax.numpy as jp

# Brax currently calls this removed JAX helper; restore its one-device semantics
# without changing PPO or the environment. This keeps CUDA and CPU backends usable.
def _device_put_replicated(tree, devices):
    return jax.tree.map(
        lambda leaf: jax.device_put(jp.stack([leaf] * len(devices)), devices[0]), tree
    )
jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from mujoco_playground._src import wrapper
from ascento.balance import AscentoBalance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    backend = jax.default_backend()
    if os.environ.get("ASCENTO_JAX_PLATFORM", "cuda") == "cuda" and backend != "gpu":
        raise RuntimeError(
            f"CUDA requested but JAX selected {backend}; install a CUDA-enabled "
            "jaxlib before running the default command."
        )
    env = AscentoBalance()
    smoke = bool(os.environ.get("ASCENTO_PPO_SMOKE_NET"))
    hidden_sizes = (64, 64) if smoke else (512, 256, 128)
    updates_per_batch = 1 if smoke else 4
    train_env = wrapper.wrap_for_brax_training(
        env, episode_length=600, action_repeat=1, full_reset=False
    )
    def network_factory(obs_size, action_size, preprocess_observations_fn):
        return ppo_networks.make_ppo_networks(
            obs_size, action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=hidden_sizes,
            value_hidden_layer_sizes=hidden_sizes,
            policy_obs_key="state", value_obs_key="state",
        )
    def progress(step, metrics):
        keys = ("episode/walltime", "eval/episode_reward",
                "training/entropy", "training/loss")
        shown = {k: float(metrics[k]) for k in keys if k in metrics}
        print(f"ppo_step={step} backend={backend} metrics={shown}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    make_inference_fn, params, metrics = ppo_train.train(
        environment=train_env,
        num_timesteps=args.timesteps,
        wrap_env=False,
        num_envs=args.num_envs,
        episode_length=600,
        action_repeat=1,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=updates_per_batch,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        network_factory=network_factory,
        seed=args.seed,
        num_evals=0,
        run_evals=False,
        progress_fn=progress,
    )
    import pickle
    with (args.output / "policy_params.pkl").open("wb") as f:
        pickle.dump(params, f)
    (args.output / "training_manifest.json").write_text(json.dumps({
        "algorithm": "brax.training.agents.ppo.train",
        "teacher": False, "pd_seed": False, "restore_params": False,
        "reset": "exact nominal upright qpos/qvel",
        "action": "six normalized direct motor efforts",
        "backend": backend, "jax_devices": [str(d) for d in jax.devices()],
        "timesteps": args.timesteps, "num_envs": args.num_envs,
        "seed": args.seed, "network_hidden_layer_sizes": list(hidden_sizes),
        "num_updates_per_batch": updates_per_batch,
        "metrics": {k: float(v) for k, v in metrics.items()
                                        if hasattr(v, "item")},
    }, indent=2), encoding="utf-8")
    print("PPO_TRAINING_COMPLETE", flush=True)

if __name__ == "__main__":
    main()

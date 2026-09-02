"""Stage-configurable Brax PPO training for the direct-torque Ascento tasks."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path


def configure_backend():
    requested = os.environ.get("ASCENTO_JAX_PLATFORM", "cuda").lower()
    if requested not in ("cuda", "cpu"):
        raise ValueError("ASCENTO_JAX_PLATFORM must be cuda or cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if requested == "cuda" else "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cuda" if requested == "cuda" else "cpu")
    # RTX 30-series GPUs use TF32 by default.  Full float32 matmuls are needed
    # for stable MJX/Brax PPO optimization on Ampere hardware.
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")


configure_backend()
import jax
import jax.numpy as jp
from brax.training.agents.ppo import train as ppo_train

from mujoco_playground._src import wrapper
from scripts.write_run_provenance import collect_provenance
from training.ppo_config import build_environment, default_ppo_kwargs, network_factory, stage_manifest
from training import safe_gradients
from training import stable_ppo_loss


def _device_put_replicated(tree, devices):
    """Compatibility shim for Brax 0.13 on JAX 0.6."""
    return jax.tree.map(lambda leaf: jax.device_put(jp.stack([leaf] * len(devices)), devices[0]), tree)


jax.device_put_replicated = _device_put_replicated
# Patch only this project's PPO invocation; installed Brax remains unmodified.
ppo_train.gradients.gradient_update_fn = safe_gradients.gradient_update_fn
ppo_train.ppo_losses.compute_ppo_loss = stable_ppo_loss.compute_ppo_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="balance")
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--unroll-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-length", type=int, default=600)
    parser.add_argument(
        "--telemetry-intervals",
        type=int,
        default=20,
        help="number of periodic PPO progress reports (0 emits only completion output)",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--init-policy", type=Path, help="previous-stage policy_params.pkl; resets optimizer")
    parser.add_argument("--resume-checkpoint", type=Path, help="Brax checkpoint directory for same-stage continuation")
    parser.add_argument("--smoke", action="store_true", help="small network/update settings for validation")
    args = parser.parse_args()
    if args.telemetry_intervals < 0:
        parser.error("--telemetry-intervals must be non-negative")

    backend = jax.default_backend()
    if os.environ.get("ASCENTO_JAX_PLATFORM", "cuda") == "cuda" and backend != "gpu":
        raise RuntimeError(f"CUDA requested but JAX selected {backend}")
    env, stage = build_environment(args.stage, args.episode_length)
    train_env = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length, action_repeat=1, full_reset=True)
    ppo_kwargs = default_ppo_kwargs(stage, args.smoke)
    for name in ("batch_size", "num_minibatches", "unroll_length"):
        value = getattr(args, name)
        if value is not None:
            ppo_kwargs[name] = value
    # Compact networks keep the first direct-torque PPO updates well-scaled on
    # a single consumer GPU.  Larger policies repeatedly produced non-finite
    # value/policy heads before the balance curriculum had stabilized.
    hidden = (64, 64) if args.smoke else (128, 128)
    restore_params = None
    if args.init_policy is not None:
        with args.init_policy.open("rb") as handle:
            restore_params = pickle.load(handle)
    args.output.mkdir(parents=True, exist_ok=True)

    training_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    provenance = collect_provenance(
        Path(__file__).resolve().parents[1],
        stage=stage.name,
        training_arguments=training_arguments,
    )
    (args.output / "source_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    telemetry_path = args.output / "telemetry.jsonl"
    started_at = time.perf_counter()

    def _scalar_metrics(metrics):
        return {
            key: float(value)
            for key, value in metrics.items()
            if hasattr(value, "item")
        }

    def progress(step, metrics):
        elapsed_seconds = time.perf_counter() - started_at
        completed_steps = min(int(step), args.timesteps)
        fraction_complete = completed_steps / args.timesteps if args.timesteps else 1.0
        steps_per_second = completed_steps / elapsed_seconds if elapsed_seconds else 0.0
        eta_seconds = (
            (args.timesteps - completed_steps) / steps_per_second
            if steps_per_second > 0
            else None
        )
        record = {
            "stage": stage.name,
            "backend": backend,
            "completed_steps": completed_steps,
            "total_steps": args.timesteps,
            "percent_complete": round(100.0 * fraction_complete, 2),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "steps_per_second": round(steps_per_second, 1),
            "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "metrics": _scalar_metrics(metrics),
        }
        with telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        eta_text = "unknown" if eta_seconds is None else f"{eta_seconds / 60:.1f} min"
        print(
            "PPO_PROGRESS "
            f"stage={stage.name} progress={record['percent_complete']:.2f}% "
            f"steps={completed_steps:,}/{args.timesteps:,} "
            f"throughput={steps_per_second:,.1f} steps/s eta={eta_text} "
            f"metrics={record['metrics']}",
            flush=True,
        )
        non_finite_metrics = [
            name for name, value in record["metrics"].items()
            if not math.isfinite(value)
        ]
        if non_finite_metrics:
            raise FloatingPointError(
                "Non-finite PPO metrics detected: " + ", ".join(non_finite_metrics)
            )

    make_inference_fn, params, metrics = ppo_train.train(
        environment=train_env,
        num_timesteps=args.timesteps,
        wrap_env=False,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=1,
        network_factory=network_factory(hidden, stage.initial_noise_std),
        seed=args.seed,
        # Keep PPO loss telemetry, but also run deterministic episodes so an
        # apparently converged optimizer cannot mask repeated physical falls.
        num_evals=args.telemetry_intervals + 1,
        run_evals=True,
        num_eval_envs=64,
        deterministic_eval=True,
        eval_env=train_env,
        progress_fn=progress,
        restore_params=restore_params,
        restore_checkpoint_path=str(args.resume_checkpoint) if args.resume_checkpoint else None,
        # Orbax requires absolute checkpoint paths when it writes asynchronously.
        save_checkpoint_path=str((args.output / "checkpoint").resolve()),
        **ppo_kwargs,
    )
    del make_inference_fn
    with (args.output / "policy_params.pkl").open("wb") as handle:
        pickle.dump(params, handle)
    manifest = {
        "algorithm": "brax.training.agents.ppo.train",
        "stage": stage_manifest(stage),
        "teacher": False,
        "pd_seed": False,
        "restore_params": args.init_policy is not None,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "action": "six normalized direct motor efforts",
        "observation_size": env.observation_size,
        "backend": backend,
        "jax_devices": [str(device) for device in jax.devices()],
        "timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "network_hidden_layer_sizes": list(hidden),
        "initial_noise_std": stage.initial_noise_std,
        "entropy_cost": stage.entropy_cost,
        "metrics": {key: float(value) for key, value in metrics.items() if hasattr(value, "item")},
        "provenance": provenance,
    }
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PPO_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

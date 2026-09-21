"""Create a fresh task checkpoint from verified compatible actor weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import ascento_mjlab.tasks  # noqa: F401


def initialize_transfer(
    *,
    source_checkpoint: Path,
    target_task: str,
    output_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    """Write a target-task checkpoint with actor-only compatible transfer."""
    source = source_checkpoint.expanduser().resolve()
    output = output_checkpoint.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing transfer checkpoint: {output}")

    cfg = load_env_cfg(target_task, play=False)
    cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg, device=device), clip_actions=load_rl_cfg(target_task).clip_actions
    )
    try:
        runner_cls = load_runner_cls(target_task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(load_rl_cfg(target_task)), device=device)
        initializer = getattr(runner, "initialize_from_compatible_actor", None)
        if not callable(initializer):
            raise TypeError(
                f"task {target_task} does not use a runner with compatible actor transfer support"
            )
        lineage = initializer(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        runner.save(str(output), infos={"lineage": lineage})
        return {
            "checkpoint": str(output),
            "target_task": target_task,
            "source_checkpoint": lineage["source_checkpoint"],
            "source_checkpoint_sha256": lineage["source_checkpoint_sha256"],
            "actor_compatibility": lineage["actor_compatibility"],
            "copied": lineage["copied"],
            "fresh": lineage["fresh"],
        }
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="verified source checkpoint")
    parser.add_argument("--task", required=True, help="new target task ID")
    parser.add_argument("--output", type=Path, required=True, help="new checkpoint path")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = initialize_transfer(
        source_checkpoint=args.source,
        target_task=args.task,
        output_checkpoint=args.output,
        device=args.device,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

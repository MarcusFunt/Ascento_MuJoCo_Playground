"""Render exactly one preview from the newest available PPO checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from training.ppo_config import build_environment
from tools.render_training_progress import (
    load_checkpoint_policy,
    numeric_checkpoints,
    render_preview,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="balance")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--initial-noise-std", type=float)
    args = parser.parse_args()

    checkpoints = numeric_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        parser.error(f"no numeric checkpoints in {args.checkpoint_dir}")
    checkpoint_path = checkpoints[-1]
    hidden_sizes = tuple(int(value) for value in args.hidden_sizes.split(",") if value)
    env, stage = build_environment(args.stage, episode_length=args.steps)
    initial_noise_std = args.initial_noise_std or stage.initial_noise_std
    policy = load_checkpoint_policy(checkpoint_path, env, hidden_sizes, initial_noise_std)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.stage}_step_{int(checkpoint_path.name):09d}.png"
    report = render_preview(env, policy, args.seed, output, args.steps)
    record = {
        "checkpoint": str(checkpoint_path.resolve()),
        "step": int(checkpoint_path.name),
        **report,
    }
    with (args.output_dir / "progress_renders.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print("PROGRESS_RENDER " + json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

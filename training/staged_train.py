"""Runs stages in order and advances only after deterministic physical checks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ascento.curriculum import STAGES, accepts_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "staged_artifacts")
    parser.add_argument("--timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--start", default="balance")
    parser.add_argument("--stop", default="unified_fine_tune")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = [stage.name for stage in STAGES]
    selected = STAGES[names.index(args.start):names.index(args.stop) + 1]
    previous = None
    for stage in selected:
        output = args.output_root / stage.name
        command = [sys.executable, "-m", "training.train", "--stage", stage.name, "--timesteps", str(args.timesteps), "--num-envs", str(args.num_envs), "--output", str(output)]
        if previous is not None:
            command.extend(["--init-policy", str(previous)])
        if args.smoke:
            command.append("--smoke")
        print(" ".join(command), flush=True)
        if args.dry_run:
            previous = output / "policy_params.pkl"
            continue
        subprocess.run(command, check=True, env=dict(os.environ))
        benchmark = [sys.executable, "-m", "evaluation.evaluate", "--stage", stage.name, "--artifact", str(output), "--output", str(output / "evaluation.json")]
        subprocess.run(benchmark, check=True, env=dict(os.environ))
        metrics = json.loads((output / "evaluation.json").read_text())["metrics"]
        if not accepts_stage(stage.name, metrics):
            raise RuntimeError(f"stage {stage.name} did not meet physical acceptance criteria: {metrics}")
        previous = output / "policy_params.pkl"


if __name__ == "__main__":
    main()

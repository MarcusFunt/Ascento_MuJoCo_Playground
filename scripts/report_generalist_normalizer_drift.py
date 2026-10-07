#!/usr/bin/env python3
"""Report actor target-error normalizer drift against a transfer checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
from pathlib import Path

import torch


def _checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _target_error_slice() -> tuple[int, int]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import ascento_mjlab.tasks  # noqa: F401
    from ascento_mjlab.normalizer_diagnostics import observation_term_slice

    cfg = load_env_cfg("Ascento-Generalist-Locomotion-Flat")
    cfg.scene.num_envs = 1
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        names = env.observation_manager.active_terms["actor"]
        dimensions = env.observation_manager.group_obs_term_dim["actor"]
        return observation_term_slice(names, dimensions, "world_target_error")
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("checkpoints", nargs="+", help="checkpoint files or glob patterns")
    args = parser.parse_args()

    from ascento_mjlab.normalizer_diagnostics import (
        compare_target_error_normalizers,
        target_error_normalizer_summary,
    )

    checkpoint_paths = sorted(
        {Path(match) for pattern in args.checkpoints for match in glob.glob(pattern)}
    )
    if not checkpoint_paths:
        parser.error("no checkpoint files matched")
    missing = [path for path in checkpoint_paths if not path.is_file()]
    if missing:
        parser.error(f"checkpoint does not exist: {missing[0]}")

    indices = _target_error_slice()
    reference = _checkpoint(args.reference)
    report = {
        "reference_checkpoint": str(args.reference.resolve()),
        "target_error_indices": list(indices),
        "preserved_transfer_statistics": target_error_normalizer_summary(reference, indices),
        "checkpoints": [],
    }
    for path in checkpoint_paths:
        candidate = _checkpoint(path)
        report["checkpoints"].append(
            {
                "checkpoint": str(path.resolve()),
                **compare_target_error_normalizers(reference, candidate, indices),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output} for {len(checkpoint_paths)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

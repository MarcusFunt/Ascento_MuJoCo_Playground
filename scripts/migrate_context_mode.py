#!/usr/bin/env python3
"""Expand a semantic-normalization checkpoint with a zero-initialized mode bit."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

from ascento_mjlab.semantic_normalization import migrate_context_mode_checkpoint_payload
from ascento_mjlab.task_contract import current_task_contract_for_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="iteration-zero semantic-normalization v1 checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new 42D context checkpoint path"
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        parser.error("source and output checkpoint paths must differ")
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if output.exists():
        parser.error(f"output already exists: {output}")

    import ascento_mjlab.tasks  # noqa: F401 - ensure task contracts are registered

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        parser.error("source checkpoint is not a supported RSL-RL checkpoint")
    target_contract = current_task_contract_for_task("Ascento-Generalist-Locomotion-Flat")
    migrated = migrate_context_mode_checkpoint_payload(
        payload, target_task_contract=target_contract
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    migrated["infos"]["context_mode_migration"] = {
        "schema": "ascento_context_mode_migration/v1",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha,
        "method": "zero_initialized_obstacle_mode_input_after_41_actor_channels",
        "actor_input_width": [41, 42],
        "critic_input_width": [50, 51],
        "learning_updates": 0,
        "target_task_topology_sha256": target_contract["topology_sha256"],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(migrated, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"migrated_checkpoint={output}")
    print(f"source_sha256={source_sha}")
    print("normalizer_contract=ascento_semantic_normalization/v2")
    print(f"task_topology_sha256={target_contract['topology_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

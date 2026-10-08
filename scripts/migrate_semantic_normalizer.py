#!/usr/bin/env python3
"""Create an exact semantic-normalizer migration of an iteration-zero checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

from ascento_mjlab.semantic_normalization import migrate_checkpoint_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="iteration-zero source checkpoint"
    )
    parser.add_argument("--output", type=Path, required=True, help="new migrated checkpoint path")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        parser.error("source and output checkpoint paths must differ")
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if output.exists():
        parser.error(f"output already exists: {output}")

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        parser.error("source checkpoint is not a supported RSL-RL checkpoint")
    migrated = migrate_checkpoint_payload(payload)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    migrated["infos"]["normalizer_migration"] = {
        "schema": "ascento_semantic_normalizer_migration/v1",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha,
        "method": "first_layer_affine_reparameterization",
        "learning_updates": 0,
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
    print("normalizer_contract=ascento_semantic_normalization/v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired comparison of two evaluation result databases."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from ascento_mjlab.plant_contract import plant_contracts_compatible

from .statistics import bootstrap_ci, iqm


def _metrics(path: Path) -> dict[str, dict[str, float]]:
    connection = sqlite3.connect(path / "results.sqlite")
    try:
        rows = connection.execute(
            "SELECT scenario_id, name, value FROM episode_metric WHERE value IS NOT NULL"
        ).fetchall()
        episodes = connection.execute("SELECT scenario_id, success FROM episode").fetchall()
    finally:
        connection.close()
    output: dict[str, dict[str, float]] = {}
    for scenario_id, success in episodes:
        output.setdefault(scenario_id, {})["success"] = float(success)
    for scenario_id, name, value in rows:
        output.setdefault(scenario_id, {})[name] = float(value)
    return output


def _plant_contract(path: Path) -> dict | None:
    """Read immutable plant provenance, if this evaluation was produced after P0."""
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    contract = manifest.get("plant_contract") if isinstance(manifest, dict) else None
    return contract if isinstance(contract, dict) else None


def _checkpoint_plant_contract(path: Path) -> dict | None:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    contract = manifest.get("checkpoint_plant_contract") if isinstance(manifest, dict) else None
    return contract if isinstance(contract, dict) else None


def ensure_compatible_plants(base: Path, candidate: Path) -> None:
    """Reject quantitative comparison unless both artifacts share one plant."""
    left = _plant_contract(base)
    right = _plant_contract(candidate)
    if left is None or right is None:
        raise ValueError(
            "cannot compare legacy evaluation artifacts without plant_contract provenance; "
            "re-evaluate both checkpoints against the current plant"
        )
    if not plant_contracts_compatible(left, right):
        raise ValueError(
            "cannot compare evaluations produced against different plant contracts; "
            "re-evaluate against a single compiled plant"
        )
    left_checkpoint = _checkpoint_plant_contract(base)
    right_checkpoint = _checkpoint_plant_contract(candidate)
    if left_checkpoint is None or right_checkpoint is None:
        raise ValueError(
            "cannot compare checkpoints with legacy plant provenance; "
            "retrain or re-establish a baseline on the current plant"
        )
    if not plant_contracts_compatible(left_checkpoint, right_checkpoint):
        raise ValueError(
            "cannot compare checkpoints trained against different plant contracts; "
            "re-establish both baselines on the current plant"
        )


def compare(base: Path, candidate: Path) -> dict:
    ensure_compatible_plants(base, candidate)
    left = _metrics(base)
    right = _metrics(candidate)
    common = sorted(set(left) & set(right))
    metrics = (
        sorted(set.intersection(*(set(left[sid]) & set(right[sid]) for sid in common)))
        if common
        else []
    )
    output = {"paired_scenarios": len(common), "metrics": {}}
    for metric_index, metric in enumerate(metrics):
        delta = np.asarray(
            [right[sid][metric] - left[sid][metric] for sid in common],
            dtype=float,
        )
        delta = delta[np.isfinite(delta)]
        if delta.size == 0:
            continue
        low, high = bootstrap_ci(delta, seed=metric_index)
        output["metrics"][metric] = {
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "iqm_delta": iqm(delta),
            "iqm_ci95_low": low,
            "iqm_ci95_high": high,
            "candidate_greater_fraction": float(np.mean(delta > 0)),
            "equal_fraction": float(np.mean(delta == 0)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = compare(args.base, args.candidate)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

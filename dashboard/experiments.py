"""Read-only adapters for declared experiment plans and explicit run links."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    """Return only fields needed by experiment navigation.

    Annotated run records may carry full telemetry, giant metadata structures,
    and checkpoint details. Embedding those for every unlinked run made the
    experiment list response hundreds of megabytes.
    """
    return {
        key: run[key]
        for key in ("id", "name", "display_name", "task", "stage", "state", "stale")
        if key in run
    }


def _arm_summary(key: str, value: dict[str, Any]) -> dict[str, Any]:
    attempts = value.get("run_attempts")
    return {
        "id": key,
        "declared_status": value.get("status"),
        "observed_outcome": value.get("conclusion") or value.get("evaluation_status"),
        "run_attempt_count": len(attempts) if isinstance(attempts, list) else None,
        "run_attempts": attempts if isinstance(attempts, list) else [],
        "declared": True,
    }


def discover_experiments(root: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    resolved_root = root.expanduser().resolve()
    programs: list[dict[str, Any]] = []
    if resolved_root.is_dir():
        for path in sorted(resolved_root.glob("*.json")):
            try:
                path.resolve().relative_to(resolved_root)
            except (OSError, ValueError):
                continue
            raw = _read(path)
            if raw is None:
                continue
            program_id = raw.get("plan_id") or raw.get("experiment_id")
            if not isinstance(program_id, str) or not program_id:
                continue
            linked_runs = []
            unlinked_runs = []
            for run in runs:
                metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
                tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
                if metadata.get("experiment_id") == program_id:
                    linked_runs.append({"run": _run_summary(run), "link_source": "experiment_id"})
                elif program_id in tags:
                    linked_runs.append({"run": _run_summary(run), "link_source": "explicit_tag"})
                elif not metadata.get("experiment_id"):
                    unlinked_runs.append(_run_summary(run))
            arms = [
                _arm_summary(key, value)
                for key, value in raw.items()
                if key.startswith("exp_") and isinstance(value, dict)
            ]
            programs.append(
                {
                    "id": program_id,
                    "title": raw.get("title") or raw.get("name") or path.stem,
                    "status": raw.get("status"),
                    "target_robot": raw.get("target_robot"),
                    "baseline": raw.get("baseline"),
                    "source_path": path.name,
                    "source_schema_version": raw.get("schema_version"),
                    "arms": arms,
                    "linked_runs": linked_runs,
                    "unlinked_run_count": len(unlinked_runs),
                    "unlinked_runs": unlinked_runs[:50],
                }
            )
    return {"programs": programs, "source": "docs/experiments", "read_only": True}

"""Content-addressed, read-only evidence links for stable checkpoints."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from dashboard.evaluations import discover_evaluations


@lru_cache(maxsize=512)
def _sha256_file(path_text: str, size_bytes: int, mtime_ns: int) -> str:
    del size_bytes, mtime_ns  # These values are part of the cache key.
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_evidence(
    *,
    run_id: str,
    run_dir: Path,
    stable_checkpoints: list[dict[str, Any]],
    evaluation_root: Path,
    checkpoint: str = "latest",
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    choices = [item for item in stable_checkpoints if item.get("stable") is True]
    selected = choices[-1] if checkpoint == "latest" and choices else next(
        (item for item in choices if item.get("relative_path") == checkpoint), None
    )
    if selected is None:
        raise FileNotFoundError("stable checkpoint is not available")
    relative = selected.get("relative_path")
    if not isinstance(relative, str):
        raise ValueError("checkpoint path is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("checkpoint path escapes its run artifact directory") from error
    if not path.is_file():
        raise FileNotFoundError("stable checkpoint file is missing")
    stat = path.stat()
    digest = _sha256_file(str(path), stat.st_size, stat.st_mtime_ns)

    metadata = {}
    try:
        value = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
        metadata = value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    evaluations = [
        {
            "evaluation_id": row["evaluation_id"],
            "suite_id": row.get("suite_id"),
            "status": row["status"],
            "evidence_class": row["evidence_class"],
            "finished_at_utc": row.get("finished_at_utc"),
        }
        for row in discover_evaluations(evaluation_root)
        if row.get("checkpoint_sha256") == digest
    ]
    selected_path = metadata.get("selected_checkpoint")
    selection_status = (
        "selected" if selected_path == relative else "not_selected"
        if selected_path
        else "not_recorded"
    )
    return {
        "checkpoint_id": f"{run_id}:{relative}",
        "run_id": run_id,
        "relative_path": relative,
        "iteration": selected.get("iteration"),
        "stable": True,
        "sha256": digest,
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "selection_status": selection_status,
        "visual_review_status": "not_recorded",
        "evaluations": evaluations,
    }

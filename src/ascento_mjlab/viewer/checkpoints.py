"""Filesystem-safe checkpoint discovery for browser policy viewers."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_CHECKPOINT_NUMBER = re.compile(r"(?:model|checkpoint)[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CheckpointInfo:
    relative_path: str
    name: str
    iteration: int | None
    size_bytes: int
    modified_at: float
    age_seconds: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def checkpoint_iteration(path: str | Path) -> int | None:
    """Extract the training iteration from a conventional checkpoint name."""
    match = _CHECKPOINT_NUMBER.search(Path(path).stem)
    return int(match.group(1)) if match else None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def discover_checkpoints(
    run_dir: Path,
    *,
    stable_age_seconds: float = 2.0,
    now: float | None = None,
) -> list[CheckpointInfo]:
    """List completed local policy checkpoints below one already-resolved run."""
    root = run_dir.expanduser().resolve()
    current_time = time.time() if now is None else float(now)
    discovered: list[CheckpointInfo] = []
    if not root.is_dir():
        return discovered

    for path in root.rglob("*.pt"):
        if not path.is_file() or not (
            path.name.startswith("model_") or path.name.startswith("checkpoint_")
        ):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        age = max(0.0, current_time - stat.st_mtime)
        if stat.st_size <= 0 or age < stable_age_seconds:
            continue
        resolved = path.resolve()
        if not _inside(resolved, root):
            continue
        discovered.append(
            CheckpointInfo(
                relative_path=resolved.relative_to(root).as_posix(),
                name=resolved.name,
                iteration=checkpoint_iteration(resolved),
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
                age_seconds=age,
            )
        )

    discovered.sort(
        key=lambda item: (
            -1 if item.iteration is None else item.iteration,
            item.modified_at,
            item.relative_path,
        )
    )
    return discovered


def resolve_run_checkpoint(
    run_dir: Path,
    selection: str,
    *,
    stable_age_seconds: float = 2.0,
) -> CheckpointInfo:
    """Resolve only checkpoints discovered inside the selected run."""
    checkpoints = discover_checkpoints(run_dir, stable_age_seconds=stable_age_seconds)
    if not checkpoints:
        raise FileNotFoundError(f"no stable checkpoints found below {run_dir}")
    if selection == "latest":
        return checkpoints[-1]

    candidate = Path(selection)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("checkpoint selection must be a run-relative path")
    normalized = candidate.as_posix().lstrip("./")
    for checkpoint in checkpoints:
        if checkpoint.relative_path == normalized:
            return checkpoint
    raise FileNotFoundError(f"checkpoint is not available for this run: {selection}")

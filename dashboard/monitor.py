"""Filesystem-backed monitoring helpers for Ascento training runs."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable


ERROR_PATTERN = re.compile(
    r"(traceback|\berror\b|exception|floatingpointerror|non-finite|cuda.*fail|out of memory|oom)",
    re.IGNORECASE,
)
RUN_MARKERS = (
    "telemetry.jsonl",
    "training.log",
    "run_status.json",
    "training_manifest.json",
)


@dataclass(frozen=True)
class RunRef:
    id: str
    path: Path
    relative_path: str


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
    except OSError:
        return []
    return records[-limit:] if limit is not None else records


def tail_lines(path: Path, count: int = 400) -> list[str]:
    if count <= 0 or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.readlines()[-count:]
    except OSError:
        return []


def error_excerpt(path: Path, max_lines: int = 80) -> list[str]:
    lines = tail_lines(path, 1200)
    if not lines:
        return []
    selected: set[int] = set()
    for index, line in enumerate(lines):
        if ERROR_PATTERN.search(line):
            selected.update(range(max(0, index - 2), min(len(lines), index + 4)))
    ordered = [lines[index].rstrip("\n") for index in sorted(selected)]
    return ordered[-max_lines:]


def discover_runs(root: Path) -> list[RunRef]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []
    candidates: set[Path] = set()
    if any((root / marker).exists() for marker in RUN_MARKERS):
        candidates.add(root)
    for marker in RUN_MARKERS:
        for path in root.rglob(marker):
            candidates.add(path.parent.resolve())
    refs = []
    for path in candidates:
        if not _inside(path, root):
            continue
        relative = "." if path == root else path.relative_to(root).as_posix()
        refs.append(RunRef(sha1(relative.encode("utf-8")).hexdigest()[:12], path, relative))
    return sorted(refs, key=lambda ref: run_modified_time(ref.path), reverse=True)


def resolve_run(root: Path, run_id: str) -> RunRef:
    for ref in discover_runs(root):
        if ref.id == run_id:
            return ref
    raise KeyError(run_id)


def run_modified_time(run_dir: Path) -> float:
    mtimes = []
    for marker in RUN_MARKERS:
        path = run_dir / marker
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes, default=run_dir.stat().st_mtime if run_dir.exists() else 0.0)


def _stage_name(run_dir: Path, telemetry: dict[str, Any] | None, manifest: dict[str, Any] | None) -> str:
    if telemetry and telemetry.get("stage"):
        return str(telemetry["stage"])
    if manifest:
        stage = manifest.get("stage")
        if isinstance(stage, dict) and stage.get("name"):
            return str(stage["name"])
        if isinstance(stage, str):
            return stage
    return run_dir.name


def latest_render(run_dir: Path) -> dict[str, Any] | None:
    manifest_candidates = [
        run_dir / "renders" / "progress_renders.jsonl",
        run_dir / "progress_renders.jsonl",
    ]
    for manifest_path in manifest_candidates:
        records = load_jsonl(manifest_path, limit=1)
        if not records:
            continue
        record = records[-1]
        raw_path = record.get("path")
        if raw_path:
            path = Path(str(raw_path))
            if not path.is_file():
                path = manifest_path.parent / path.name
            if path.is_file() and _inside(path, run_dir):
                return {**record, "filename": path.name, "relative_path": path.relative_to(run_dir).as_posix()}
    pngs = list((run_dir / "renders").glob("*.png")) if (run_dir / "renders").exists() else []
    if not pngs:
        pngs = list(run_dir.glob("*.png"))
    if not pngs:
        return None
    path = max(pngs, key=lambda item: item.stat().st_mtime)
    return {"filename": path.name, "relative_path": path.relative_to(run_dir).as_posix()}


def summarize_run(run_dir: Path, root: Path, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    telemetry_records = load_jsonl(run_dir / "telemetry.jsonl", limit=1)
    telemetry = telemetry_records[-1] if telemetry_records else None
    manifest = load_json(run_dir / "training_manifest.json")
    status = load_json(run_dir / "run_status.json") or {}
    errors = error_excerpt(run_dir / "training.log")

    state = status.get("state")
    if not state:
        if manifest:
            state = "finished"
        elif errors and now - run_modified_time(run_dir) > 30:
            state = "error"
        elif telemetry:
            state = "running" if now - run_modified_time(run_dir) < 600 else "unknown"
        else:
            state = "unknown"

    relative = "." if run_dir.resolve() == root.resolve() else run_dir.resolve().relative_to(root.resolve()).as_posix()
    return {
        "id": sha1(relative.encode("utf-8")).hexdigest()[:12],
        "name": relative,
        "stage": _stage_name(run_dir, telemetry, manifest),
        "state": state,
        "modified_at": run_modified_time(run_dir),
        "status": status,
        "telemetry": telemetry,
        "errors": errors,
        "latest_render": latest_render(run_dir),
        "has_log": (run_dir / "training.log").is_file(),
        "has_checkpoint": (run_dir / "checkpoint").is_dir(),
    }


def list_run_summaries(root: Path) -> list[dict[str, Any]]:
    return [summarize_run(ref.path, root) for ref in discover_runs(root)]


def numeric_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        (path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )

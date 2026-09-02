"""Filesystem-backed monitoring helpers for Ascento training runs."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

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


def _training_limit(run_dir: Path) -> int | None:
  """Read the configured iteration count when an RSL-RL run recorded it."""
  params_path = run_dir / "params" / "agent.yaml"
  if not params_path.is_file():
    return None
  try:
    contents = params_path.read_text(encoding="utf-8")
  except OSError:
    return None
  match = re.search(r"^\s*max_iterations:\s*(\d+)\s*$", contents, re.MULTILINE)
  return int(match.group(1)) if match else None


def load_tensorboard_records(run_dir: Path, limit: int | None = 2000) -> list[dict[str, Any]]:
    """Convert native RSL-RL TensorBoard scalars to the dashboard schema."""
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return []
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    by_step: dict[int, dict[str, Any]] = {}
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 20_000})
            accumulator.Reload()
        except (KeyError, OSError, RuntimeError, ValueError):
            continue
        for tag in accumulator.Tags().get("scalars", []):
            try:
                events = accumulator.Scalars(tag)
            except (KeyError, RuntimeError):
                continue
            for event in events:
                record = by_step.setdefault(
                    int(event.step),
                    {"completed_steps": int(event.step), "metrics": {}},
                )
                record["metrics"][tag] = float(event.value)
                record["wall_time"] = float(event.wall_time)

    records = [by_step[step] for step in sorted(by_step)]
    total = _training_limit(run_dir)
    if total is not None:
        for record in records:
            completed = record["completed_steps"]
            record["total_steps"] = total
            record["percent_complete"] = min(100.0, 100.0 * completed / total)
            fps = record["metrics"].get("Perf/total_fps")
            if fps is not None and fps > 0:
                record["steps_per_second"] = fps
                record["eta_seconds"] = max(0.0, (total - completed) / fps)
    return records[-limit:] if limit is not None else records


def load_training_records(run_dir: Path, limit: int | None = 2000) -> list[dict[str, Any]]:
    """Read legacy JSON telemetry or native RSL-RL TensorBoard telemetry."""
    records = load_jsonl(run_dir / "telemetry.jsonl", limit=limit)
    return records if records else load_tensorboard_records(run_dir, limit=limit)


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
    for pattern in ("events.out.tfevents.*", "model_*.pt"):
        for path in root.rglob(pattern):
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
    for pattern in ("events.out.tfevents.*", "model_*.pt"):
        for path in run_dir.glob(pattern):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                pass
    return max(mtimes, default=run_dir.stat().st_mtime if run_dir.exists() else 0.0)


def _stage_name(
    run_dir: Path,
    telemetry: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    status: dict[str, Any] | None = None,
) -> str:
    if telemetry and telemetry.get("stage"):
        return str(telemetry["stage"])
    if manifest:
        stage = manifest.get("stage")
        if isinstance(stage, dict) and stage.get("name"):
            return str(stage["name"])
        if isinstance(stage, str):
            return stage
    if status and status.get("stage"):
        return str(status["stage"])
    experiment = run_dir.parent.name
    if experiment.startswith("ascento_"):
        return experiment.removeprefix("ascento_")
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
    telemetry_records = load_training_records(run_dir, limit=1)
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
        elif telemetry and now - run_modified_time(run_dir) < 30:
            state = "running"
        elif telemetry or (run_dir / "params" / "agent.yaml").is_file():
            state = "finished"
        else:
            state = "unknown"

    relative = "." if run_dir.resolve() == root.resolve() else run_dir.resolve().relative_to(root.resolve()).as_posix()
    return {
        "id": sha1(relative.encode("utf-8")).hexdigest()[:12],
        "name": relative,
        "stage": _stage_name(run_dir, telemetry, manifest, status),
        "state": state,
        "modified_at": run_modified_time(run_dir),
        "status": status,
        "telemetry": telemetry,
        "errors": errors,
        "latest_render": latest_render(run_dir),
        "has_log": (run_dir / "training.log").is_file(),
        "has_checkpoint": (run_dir / "checkpoint").is_dir() or bool(list(run_dir.glob("model_*.pt"))),
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

"""Run lifecycle, provenance metadata, and comparison services for the dashboard."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any
from uuid import uuid4

from dashboard.config import REPO_ROOT
from dashboard.health import resolve_dashboard_run, run_status_path, summarize_dashboard_run
from dashboard.versioning import annotate_run_summary

RUN_METADATA = "run_metadata.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "run"


def _run_id(relative_path: str) -> str:
    return sha1(relative_path.encode("utf-8")).hexdigest()[:12]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _clean_tags(tags: list[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        cleaned = str(tag).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned[:64])
    return result[:32]


class RunService:
    """Filesystem-backed run controller used by the dashboard API."""

    def __init__(self, artifact_root: Path, *, stale_after_seconds: float = 90.0):
        self.artifact_root = artifact_root.expanduser().resolve()
        self.stale_after_seconds = stale_after_seconds

    def metadata_path(self, run_dir: Path) -> Path:
        """Find launcher-level metadata even when RSL-RL artifacts are nested below it."""
        for directory in (run_dir.resolve(), *run_dir.resolve().parents):
            if not _inside(directory, self.artifact_root):
                break
            candidate = directory / RUN_METADATA
            if candidate.is_file():
                return candidate
            if directory == self.artifact_root:
                break
        return run_dir / RUN_METADATA

    def load_metadata(self, run_dir: Path) -> dict[str, Any]:
        metadata = _load_json(self.metadata_path(run_dir))
        return {
            "display_name": metadata.get("display_name") or run_dir.name,
            "notes": metadata.get("notes") or "",
            "tags": _clean_tags(metadata.get("tags") if isinstance(metadata.get("tags"), list) else []),
            "purpose": metadata.get("purpose") or "",
            "parent_run_id": metadata.get("parent_run_id"),
            "parent_checkpoint": metadata.get("parent_checkpoint"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "schema_version": int(metadata.get("schema_version") or 1),
        }

    def annotate(self, summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        metadata = self.load_metadata(run_dir)
        summary["metadata"] = metadata
        summary["display_name"] = metadata["display_name"]
        summary["notes"] = metadata["notes"]
        summary["tags"] = metadata["tags"]
        summary["lineage"] = {
            "parent_run_id": metadata.get("parent_run_id"),
            "parent_checkpoint": metadata.get("parent_checkpoint"),
        }
        return annotate_run_summary(summary, run_dir, self.artifact_root)

    def detail(self, run_id: str) -> dict[str, Any]:
        ref = resolve_dashboard_run(self.artifact_root, run_id)
        summary = summarize_dashboard_run(
            ref.path,
            self.artifact_root,
            stale_after_seconds=self.stale_after_seconds,
            detailed=True,
        )
        return self.annotate(summary, ref.path)

    def update_metadata(self, run_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        ref = resolve_dashboard_run(self.artifact_root, run_id)
        path = self.metadata_path(ref.path)
        current = _load_json(path)
        if not current:
            current = {
                "schema_version": 1,
                "created_at": _now(),
                "display_name": ref.path.name,
                "notes": "",
                "tags": [],
                "purpose": "",
                "parent_run_id": None,
                "parent_checkpoint": None,
            }

        for key in ("display_name", "notes", "purpose", "parent_checkpoint"):
            if key in changes and changes[key] is not None:
                current[key] = str(changes[key]).strip()
        if "tags" in changes and changes["tags"] is not None:
            current["tags"] = _clean_tags(changes["tags"])
        if "parent_run_id" in changes:
            parent = changes["parent_run_id"]
            if parent:
                if str(parent) == run_id:
                    raise ValueError("a run cannot be its own parent")
                resolve_dashboard_run(self.artifact_root, str(parent))
                current["parent_run_id"] = str(parent)
            else:
                current["parent_run_id"] = None
        current["updated_at"] = _now()
        _write_json(path, current)
        return self.detail(run_id)

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        display_name = str(request.get("display_name") or "").strip()
        if not display_name:
            raise ValueError("display_name is required")
        task = str(request.get("task") or "Ascento-Balance-Flat").strip()
        if not task.startswith("Ascento-"):
            raise ValueError("task must be an Ascento task name")
        training_args = request.get("training_args") or []
        if not isinstance(training_args, list) or not all(isinstance(value, str) for value in training_args):
            raise ValueError("training_args must be a list of strings")
        parent_run_id = request.get("parent_run_id")
        if parent_run_id:
            resolve_dashboard_run(self.artifact_root, str(parent_run_id))

        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if not os.access(self.artifact_root, os.W_OK | os.X_OK):
            raise PermissionError(f"artifact root is not writable: {self.artifact_root}")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = f"{stamp}_{_slug(display_name)}_{uuid4().hex[:8]}"
        command = [
            sys.executable,
            "-m",
            "dashboard.launch",
            "--artifact-root",
            str(self.artifact_root),
            "--name",
            directory_name,
            "--task",
            task,
            "--display-name",
            display_name,
        ]
        notes = str(request.get("notes") or "").strip()
        purpose = str(request.get("purpose") or "").strip()
        parent_checkpoint = str(request.get("parent_checkpoint") or "").strip()
        if notes:
            command.extend(["--notes", notes])
        if purpose:
            command.extend(["--purpose", purpose])
        if parent_run_id:
            command.extend(["--parent-run-id", str(parent_run_id)])
        if parent_checkpoint:
            command.extend(["--parent-checkpoint", parent_checkpoint])
        for tag in _clean_tags(request.get("tags")):
            command.extend(["--tag", tag])
        command.append("--")
        command.extend(training_args)

        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return {
            "id": _run_id(directory_name),
            "name": directory_name,
            "display_name": display_name,
            "task": task,
            "state": "starting",
            "launcher_pid": process.pid,
            "created_at": _now(),
        }

    def stop(self, run_id: str, *, reason: str = "user_requested") -> dict[str, Any]:
        ref = resolve_dashboard_run(self.artifact_root, run_id)
        status_path = run_status_path(ref.path, self.artifact_root)
        status = _load_json(status_path)
        state = str(status.get("state") or "unknown")
        if state not in {"starting", "running", "stopping"}:
            raise ValueError(f"run is not active (state={state})")

        status.update(
            {
                "state": "stopping",
                "stop_requested_at": _now(),
                "stop_reason": reason,
            }
        )
        _write_json(status_path, status)

        pgid = status.get("process_group")
        launcher_pid = status.get("launcher_pid")
        trainer_pid = status.get("pid")
        try:
            if isinstance(pgid, int) and pgid > 0:
                os.killpg(pgid, signal.SIGINT)
            elif isinstance(launcher_pid, int) and launcher_pid > 0:
                os.kill(launcher_pid, signal.SIGINT)
            elif isinstance(trainer_pid, int) and trainer_pid > 0:
                os.kill(trainer_pid, signal.SIGINT)
            else:
                raise ProcessLookupError("no controllable process id recorded")
        except ProcessLookupError as error:
            status.update(
                {
                    "state": "error",
                    "control_error": str(error),
                    "finished_at": _now(),
                }
            )
            _write_json(status_path, status)
            raise
        return self.detail(run_id)

    def compare(self, run_ids: list[str]) -> dict[str, Any]:
        unique = list(dict.fromkeys(run_ids))
        if len(unique) < 2:
            raise ValueError("compare requires at least two runs")
        if len(unique) > 8:
            raise ValueError("compare supports at most eight runs")

        runs = [self.detail(run_id) for run_id in unique]
        metric_names = ("reward", "episode_length", "ppo_loss", "entropy", "kl", "clip_fraction")
        baseline_latest = (runs[0].get("training_health") or {}).get("latest") or {}
        comparison: list[dict[str, Any]] = []
        for run in runs:
            latest = (run.get("training_health") or {}).get("latest") or {}
            deltas: dict[str, float | None] = {}
            for metric in metric_names:
                base = baseline_latest.get(metric)
                value = latest.get(metric)
                if isinstance(base, (int, float)) and isinstance(value, (int, float)):
                    deltas[metric] = float(value) - float(base)
                else:
                    deltas[metric] = None
            progress = run.get("telemetry") or {}
            comparison.append(
                {
                    "id": run.get("id"),
                    "display_name": run.get("display_name"),
                    "state": run.get("state"),
                    "stage": run.get("stage"),
                    "tags": run.get("tags") or [],
                    "repository_version": run.get("repository_version"),
                    "iteration": progress.get("iteration"),
                    "percent_complete": progress.get("percent_complete"),
                    "latest_metrics": {name: latest.get(name) for name in metric_names},
                    "delta_from_baseline": deltas,
                }
            )
        return {"baseline_id": unique[0], "runs": comparison}

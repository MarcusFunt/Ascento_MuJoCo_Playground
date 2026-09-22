"""Managed browser-viewer subprocesses for the Ascento dashboard."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ascento_mjlab.viewer.checkpoints import (
    CheckpointInfo,
    discover_checkpoints,
    resolve_run_checkpoint,
)
from dashboard.config import REPO_ROOT
from dashboard.run_service import RunService


class ViewerBusyError(RuntimeError):
    """Raised when the single managed viewer slot is already occupied."""


class ViewerNotFoundError(KeyError):
    """Raised when a viewer ID is no longer known."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


@dataclass
class _ManagedViewer:
    viewer_id: str
    run_id: str
    task: str
    port: int
    follow: bool
    selected_checkpoint: CheckpointInfo
    process: subprocess.Popen
    log_path: Path
    status_path: Path
    started_at: str
    state: str = "starting"
    exit_code: int | None = None


class ViewerService:
    """Own a disposable viewer process without coupling it to the trainer."""

    def __init__(
        self,
        run_service: RunService,
        *,
        logs_root: Path,
        host: str = "0.0.0.0",
        port: int = 8081,
        stable_age_seconds: float = 2.0,
    ) -> None:
        self.run_service = run_service
        self.logs_root = logs_root.expanduser().resolve()
        self.host = host
        self.port = int(port)
        self.stable_age_seconds = max(0.0, float(stable_age_seconds))
        self._lock = threading.RLock()
        self._active: _ManagedViewer | None = None

    def checkpoints(self, run_id: str) -> dict[str, Any]:
        ref = self.run_service.resolve(run_id)
        items = discover_checkpoints(
            ref.path,
            stable_age_seconds=self.stable_age_seconds,
        )
        return {
            "run_id": run_id,
            "latest": items[-1].relative_path if items else None,
            "checkpoints": [item.as_dict() for item in items],
        }

    def start(
        self,
        *,
        run_id: str,
        checkpoint: str = "latest",
        follow: bool = False,
        device: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active is not None:
                self._refresh_locked(self._active)
                if self._active.state in {"starting", "running", "stopping"}:
                    raise ViewerBusyError(
                        "a viewer is already active; stop it before starting another"
                    )

            ref = self.run_service.resolve(run_id)
            detail = self.run_service.detail(run_id)
            run_info = detail.get("run_info")
            run_info = run_info if isinstance(run_info, dict) else {}
            task = run_info.get("task")
            if not isinstance(task, str) or not task.startswith("Ascento-"):
                raise ValueError("selected run does not expose a valid Ascento task")

            selected = resolve_run_checkpoint(
                ref.path,
                checkpoint,
                stable_age_seconds=self.stable_age_seconds,
            )
            viewer_id = uuid4().hex[:12]
            self.logs_root.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_root / f"{viewer_id}.log"
            status_path = self.logs_root / f"{viewer_id}.json"
            command = [
                sys.executable,
                "-m",
                "ascento_mjlab.viewer.worker",
                "--task",
                task,
                "--run-dir",
                str(ref.path),
                "--checkpoint",
                selected.relative_path,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--status-file",
                str(status_path),
                "--stable-age-seconds",
                str(self.stable_age_seconds),
            ]
            if follow:
                command.append("--follow")
            if device:
                command.extend(["--device", str(device)])

            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )

            self._active = _ManagedViewer(
                viewer_id=viewer_id,
                run_id=run_id,
                task=task,
                port=self.port,
                follow=bool(follow),
                selected_checkpoint=selected,
                process=process,
                log_path=log_path,
                status_path=status_path,
                started_at=_now(),
            )
            return self._snapshot_locked(self._active)

    def list(self) -> dict[str, Any]:
        with self._lock:
            if self._active is None:
                return {"viewers": []}
            return {"viewers": [self._snapshot_locked(self._active)]}

    def get(self, viewer_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            return self._snapshot_locked(viewer)

    def logs(self, viewer_id: str, *, tail: int = 300) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            limit = max(1, min(int(tail), 5000))
            try:
                lines = viewer.log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                lines = []
            return {"viewer_id": viewer_id, "lines": lines[-limit:]}

    def stop(self, viewer_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            self._refresh_locked(viewer)
            if viewer.state not in {"starting", "running", "stopping"}:
                return self._snapshot_locked(viewer)
            viewer.state = "stopping"
            try:
                os.killpg(viewer.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                try:
                    viewer.process.terminate()
                except OSError:
                    pass
            return self._snapshot_locked(viewer)

    def stop_all(self) -> None:
        with self._lock:
            if self._active is None:
                return
            viewer = self._active
            self._refresh_locked(viewer)
            if viewer.state not in {"starting", "running", "stopping"}:
                return
            try:
                os.killpg(viewer.process.pid, signal.SIGTERM)
            except OSError:
                try:
                    viewer.process.terminate()
                except OSError:
                    pass
            viewer.state = "stopping"

    def _require(self, viewer_id: str) -> _ManagedViewer:
        if self._active is None or self._active.viewer_id != viewer_id:
            raise ViewerNotFoundError(viewer_id)
        return self._active

    def _refresh_locked(self, viewer: _ManagedViewer) -> None:
        exit_code = viewer.process.poll()
        if exit_code is not None:
            viewer.exit_code = int(exit_code)
            if viewer.state == "stopping" or exit_code == 0:
                viewer.state = "stopped"
            else:
                viewer.state = "failed"
            return
        if viewer.state == "stopping":
            return
        viewer.state = "running" if self._port_open() else "starting"

    def _port_open(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.08):
                return True
        except OSError:
            return False

    def _snapshot_locked(self, viewer: _ManagedViewer) -> dict[str, Any]:
        self._refresh_locked(viewer)
        runtime = _read_json(viewer.status_path)
        checkpoint = runtime.get("checkpoint") or viewer.selected_checkpoint.relative_path
        checkpoint_iteration = runtime.get("checkpoint_iteration")
        if not isinstance(checkpoint_iteration, int):
            checkpoint_iteration = viewer.selected_checkpoint.iteration

        training_iteration: int | None = None
        try:
            progress = self.run_service.progress(viewer.run_id)
            telemetry = progress.get("telemetry")
            telemetry = telemetry if isinstance(telemetry, dict) else {}
            value = telemetry.get("iteration")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                training_iteration = int(value)
        except (KeyError, OSError, ValueError):
            pass

        lag_iterations: int | None = None
        if training_iteration is not None and checkpoint_iteration is not None:
            lag_iterations = max(0, training_iteration - checkpoint_iteration)

        return {
            "id": viewer.viewer_id,
            "run_id": viewer.run_id,
            "task": viewer.task,
            "state": viewer.state,
            "pid": viewer.process.pid,
            "port": viewer.port,
            "follow": viewer.follow,
            "checkpoint": checkpoint,
            "checkpoint_iteration": checkpoint_iteration,
            "training_iteration": training_iteration,
            "lag_iterations": lag_iterations,
            "started_at": viewer.started_at,
            "loaded_at": runtime.get("loaded_at"),
            "exit_code": viewer.exit_code,
        }

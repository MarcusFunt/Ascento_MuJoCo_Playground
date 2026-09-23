"""Managed browser-viewer subprocesses for the Ascento dashboard."""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
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
from ascento_mjlab.viewer.ipc import IntrospectionIPC
from ascento_mjlab.viewer.replay import PolicyReplayRecorder
from dashboard.config import REPO_ROOT
from dashboard.policy_architecture import inspect_policy_checkpoint
from dashboard.run_service import RunService

_FORCE_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


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


def _finite_vector(value: Any, expected: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == expected
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


@dataclass
class _ManagedViewer:
    viewer_id: str
    run_id: str
    task: str
    port: int
    follow: bool
    jacobian_hz: float
    selected_checkpoint: CheckpointInfo
    process: subprocess.Popen
    log_path: Path
    status_path: Path
    introspection_dir: Path
    capture_root: Path
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
        stop_grace_seconds: float = 5.0,
        stop_term_seconds: float = 5.0,
    ) -> None:
        self.run_service = run_service
        self.logs_root = logs_root.expanduser().resolve()
        self.host = host
        self.port = int(port)
        self.stable_age_seconds = max(0.0, float(stable_age_seconds))
        self.stop_grace_seconds = max(0.0, float(stop_grace_seconds))
        self.stop_term_seconds = max(0.0, float(stop_term_seconds))
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

    def architecture(self, run_id: str) -> dict[str, Any]:
        """Describe the newest stable policy without exposing checkpoint values."""
        ref = self.run_service.resolve(run_id)
        items = discover_checkpoints(
            ref.path,
            stable_age_seconds=self.stable_age_seconds,
        )
        if not items:
            return {"available": False, "message": "No stable checkpoint is available yet."}

        selected = items[-1]
        try:
            return inspect_policy_checkpoint(
                ref.path / selected.relative_path,
                selected.relative_path,
            )
        except ValueError as error:
            return {
                "available": False,
                "checkpoint": selected.relative_path,
                "iteration": selected.iteration,
                "message": str(error),
            }

    def start(
        self,
        *,
        run_id: str,
        checkpoint: str = "latest",
        follow: bool = False,
        device: str | None = None,
        jacobian_hz: float = 2.0,
    ) -> dict[str, Any]:
        if float(jacobian_hz) not in (0.0, 1.0, 2.0, 5.0, 10.0):
            raise ValueError("Jacobian frequency must be one of 0, 1, 2, 5, 10 Hz")
        with self._lock:
            if self._active is not None:
                self._refresh_locked(self._active)
                if self._active.state in {"starting", "running", "stopping"}:
                    raise ViewerBusyError(
                        "a viewer is already active; stop it before starting another"
                    )
            if self._port_open():
                raise ViewerBusyError(
                    f"viewer port {self.port} is already in use"
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
            introspection_dir = self.logs_root / viewer_id / "introspection"
            introspection_dir.mkdir(parents=True, exist_ok=True)
            capture_root = ref.path / "viewer_diagnostics" / viewer_id / "events"
            capture_root.mkdir(parents=True, exist_ok=True)
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
                "--jacobian-hz",
                str(float(jacobian_hz)),
                "--introspection-dir",
                str(introspection_dir),
                "--viewer-id",
                viewer_id,
                "--run-id",
                run_id,
                "--capture-dir",
                str(capture_root),
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
                jacobian_hz=float(jacobian_hz),
                selected_checkpoint=selected,
                process=process,
                log_path=log_path,
                status_path=status_path,
                introspection_dir=introspection_dir,
                capture_root=capture_root,
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

    def introspection_schema(self, viewer_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            value = _read_json(viewer.introspection_dir / "schema.json")
            return value or {"available": False, "message": "Viewer schema is not ready yet."}

    def introspection_latest(self, viewer_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            value = _read_json(viewer.introspection_dir / "latest.json")
            return value or {"available": False, "message": "No viewer frame is available yet."}

    def introspection_captures(self, viewer_id: str, *, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            recorder = PolicyReplayRecorder(
                viewer.capture_root,
                viewer_id=viewer_id,
                run_id=viewer.run_id,
            )
            captures = recorder.list_summaries(limit=max(1, min(int(limit), 200)))
            return {
                "viewer_id": viewer_id,
                "captures": [
                    {key: value for key, value in item.items() if key not in {"frames", "schema"}}
                    for item in captures
                ],
            }

    def introspection_capture(self, viewer_id: str, event_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            recorder = PolicyReplayRecorder(
                viewer.capture_root,
                viewer_id=viewer_id,
                run_id=viewer.run_id,
            )
            value = recorder.read_event(event_id)
            if value is None:
                raise FileNotFoundError(event_id)
            return value

    def request_introspection_capture(self, viewer_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            self._refresh_locked(viewer)
            if viewer.state not in {"starting", "running"}:
                raise ViewerBusyError("viewer is not running")
            request_id = IntrospectionIPC(viewer.introspection_dir).request_manual_capture()
            return {"viewer_id": viewer_id, "request_id": request_id, "state": "queued"}

    def request_introspection_explanation(
        self,
        viewer_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            self._refresh_locked(viewer)
            if viewer.state not in {"starting", "running"}:
                raise ViewerBusyError("viewer is not running")
            schema = _read_json(viewer.introspection_dir / "schema.json")
            latest = _read_json(viewer.introspection_dir / "latest.json")
            actor_schema = schema.get("actor") if isinstance(schema, dict) else None
            observation_dim = actor_schema.get("input_dim") if isinstance(actor_schema, dict) else None
            if not isinstance(observation_dim, int) or observation_dim <= 0:
                raise ValueError("viewer introspection schema is not ready")
            checkpoint = str(payload.get("checkpoint") or "")
            if not isinstance(schema, dict) or checkpoint != schema.get("checkpoint"):
                raise ValueError("explanations require the checkpoint currently loaded in the viewer")
            input_raw = payload.get("input_raw")
            if not _finite_vector(input_raw, observation_dim):
                raise ValueError(f"input_raw must contain {observation_dim} finite values")
            action_index = payload.get("action_index")
            latest_actions = latest.get("actor_output") if isinstance(latest, dict) else None
            if (
                not isinstance(action_index, int)
                or isinstance(action_index, bool)
                or not isinstance(latest_actions, list)
                or not 0 <= action_index < len(latest_actions)
            ):
                raise ValueError("action_index is outside the current viewer action dimension")
            baseline_kind = payload.get("baseline_kind")
            if baseline_kind not in {"normalizer_mean", "episode_start", "selected_frame"}:
                raise ValueError("unsupported explanation baseline")
            if baseline_kind == "selected_frame" and not _finite_vector(
                payload.get("baseline_raw"), observation_dim
            ):
                raise ValueError(f"selected-frame baseline must contain {observation_dim} finite values")
            n_steps = payload.get("n_steps", 32)
            if not isinstance(n_steps, int) or isinstance(n_steps, bool) or not 8 <= n_steps <= 512:
                raise ValueError("n_steps must be an integer between 8 and 512")
            request = {
                "checkpoint": checkpoint,
                "input_raw": input_raw,
                "action_index": action_index,
                "action_name": str(payload.get("action_name") or f"action_{action_index}"),
                "baseline_kind": baseline_kind,
                "baseline_raw": payload.get("baseline_raw"),
                "baseline_description": str(payload.get("baseline_description") or ""),
                "n_steps": n_steps,
                "live_sequence": payload.get("live_sequence"),
                "episode_id": payload.get("episode_id"),
                "event_id": payload.get("event_id"),
            }
            request_id = IntrospectionIPC(viewer.introspection_dir).queue_explanation_request(request)
            return {"viewer_id": viewer_id, "explanation_id": request_id, "state": "queued"}

    def introspection_explanation(self, viewer_id: str, explanation_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            value = IntrospectionIPC(viewer.introspection_dir).read_explanation(explanation_id)
            return value or {"explanation_id": explanation_id, "status": "pending"}

    def stop(self, viewer_id: str) -> dict[str, Any]:
        with self._lock:
            viewer = self._require(viewer_id)
            self._refresh_locked(viewer)
            if viewer.state not in {"starting", "running", "stopping"}:
                return self._snapshot_locked(viewer)
            self._begin_stop_locked(viewer)
            return self._snapshot_locked(viewer)

    def stop_all(self) -> None:
        with self._lock:
            if self._active is None:
                return
            viewer = self._active
            self._refresh_locked(viewer)
            if viewer.state not in {"starting", "running", "stopping"}:
                return
            self._begin_stop_locked(viewer)

    def _begin_stop_locked(self, viewer: _ManagedViewer) -> None:
        if viewer.state == "stopping":
            return
        viewer.state = "stopping"
        self._signal_process_group(viewer, signal.SIGINT)
        threading.Thread(
            target=self._escalate_stop,
            args=(viewer.viewer_id, viewer.process.pid),
            daemon=True,
            name=f"viewer-stop-{viewer.viewer_id}",
        ).start()

    def _signal_process_group(self, viewer: _ManagedViewer, sig: signal.Signals) -> None:
        killpg = getattr(os, "killpg", None)
        if callable(killpg):
            try:
                killpg(viewer.process.pid, sig)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        try:
            if sig == _FORCE_KILL_SIGNAL:
                viewer.process.kill()
            else:
                viewer.process.terminate()
        except OSError:
            pass

    def _escalate_stop(self, viewer_id: str, pid: int) -> None:
        for delay, sig in (
            (self.stop_grace_seconds, signal.SIGTERM),
            (self.stop_term_seconds, _FORCE_KILL_SIGNAL),
        ):
            time.sleep(delay)
            with self._lock:
                viewer = self._active
                if (
                    viewer is None
                    or viewer.viewer_id != viewer_id
                    or viewer.process.pid != pid
                    or viewer.state != "stopping"
                    or viewer.process.poll() is not None
                ):
                    return
                self._signal_process_group(viewer, sig)

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
            "jacobian_hz": viewer.jacobian_hz,
            "checkpoint": checkpoint,
            "checkpoint_iteration": checkpoint_iteration,
            "training_iteration": training_iteration,
            "lag_iterations": lag_iterations,
            "started_at": viewer.started_at,
            "loaded_at": runtime.get("loaded_at"),
            "exit_code": viewer.exit_code,
            "introspection_available": (viewer.introspection_dir / "schema.json").is_file(),
            "schema_version": (
                _read_json(viewer.introspection_dir / "schema.json").get("schema_version")
            ),
            "latest_sequence": (
                _read_json(viewer.introspection_dir / "latest.json").get("sequence_id")
            ),
            "last_frame_at": (
                _read_json(viewer.introspection_dir / "latest.json").get("captured_at")
            ),
            "capture_count": sum(1 for _ in viewer.capture_root.glob("event_*.json.gz")),
        }

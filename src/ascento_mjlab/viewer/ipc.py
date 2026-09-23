"""Atomic file IPC between the isolated viewer and the dashboard process."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


class ExplanationBusyError(RuntimeError):
    """An on-demand explanation already occupies this viewer's single work slot."""


class CaptureBusyError(RuntimeError):
    """A manual replay capture request is already queued for this viewer."""


class IntrospectionIPC:
    """Publish a bounded live snapshot without coupling viewer and dashboard IPC."""

    SCHEMA_VERSION = 1

    def __init__(self, directory: Path, *, publish_hz: float = 10.0) -> None:
        if publish_hz <= 0:
            raise ValueError("introspection publish frequency must be positive")
        self.directory = directory.expanduser().resolve()
        self.schema_path = self.directory / "schema.json"
        self.latest_path = self.directory / "latest.json"
        self.publish_interval = 1.0 / float(publish_hz)
        self._last_publish: float | None = None

    def publish_schema(
        self,
        actor_schema: Any,
        critic_schema: Any,
        *,
        checkpoint: str,
        policy_generation: int = 1,
        actor_network: dict[str, Any] | None = None,
        critic_network: dict[str, Any] | None = None,
    ) -> None:
        value = {
            "schema_version": self.SCHEMA_VERSION,
            "checkpoint": str(checkpoint),
            "policy_generation": int(policy_generation),
            "actor": actor_schema.to_dict(),
            "critic": critic_schema.to_dict() if critic_schema is not None else None,
            "actor_network": actor_network,
            "critic_network": critic_network,
        }
        _atomic_write_json(self.schema_path, value)

    def publish_frame(
        self,
        frame: Any,
        *,
        force: bool = False,
        now: float | None = None,
    ) -> bool:
        current = time.monotonic() if now is None else float(now)
        if (
            not force
            and self._last_publish is not None
            and current - self._last_publish < self.publish_interval
        ):
            return False
        _atomic_write_json(self.latest_path, frame.to_dict())
        self._last_publish = current
        return True

    def read_schema(self) -> dict[str, Any] | None:
        return _read_json(self.schema_path)

    def read_latest(self) -> dict[str, Any] | None:
        return _read_json(self.latest_path)

    def request_manual_capture(self, *, requested_by: str = "dashboard") -> str:
        request_id = uuid4().hex
        path = self.directory / f"capture-request-{request_id}.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.directory / "capture-slot.lock"
        try:
            with lock_path.open("x", encoding="utf-8") as lock_file:
                lock_file.write(request_id)
        except FileExistsError as error:
            raise CaptureBusyError("a manual capture is already queued") from error
        try:
            _atomic_write_json(path, {"request_id": request_id, "requested_by": requested_by})
        except Exception:
            lock_path.unlink(missing_ok=True)
            raise
        return request_id

    def consume_manual_capture_request(self) -> dict[str, Any] | None:
        requests = sorted(self.directory.glob("capture-request-*.json"))
        if not requests:
            return None
        path = requests[0]
        claimed = path.with_suffix(".processing")
        try:
            path.replace(claimed)
        except FileNotFoundError:
            return None
        try:
            value = _read_json(claimed)
            return value
        finally:
            value = locals().get("value")
            lock_path = self.directory / "capture-slot.lock"
            try:
                if isinstance(value, dict) and lock_path.read_text(encoding="utf-8") == value.get("request_id"):
                    lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            claimed.unlink(missing_ok=True)

    def queue_explanation_request(self, payload: dict[str, Any]) -> str:
        request_id = uuid4().hex
        value = {**payload, "explanation_id": request_id}
        path = self.directory / f"explanation-request-{request_id}.json"
        lock_path = self.directory / "explanation-slot.lock"
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            with lock_path.open("x", encoding="utf-8") as lock_file:
                lock_file.write(request_id)
        except FileExistsError as error:
            raise ExplanationBusyError("an explanation is already queued or running") from error
        try:
            _atomic_write_json(path, value)
        except Exception:
            lock_path.unlink(missing_ok=True)
            raise
        return request_id

    def consume_explanation_request(self) -> dict[str, Any] | None:
        requests = sorted(self.directory.glob("explanation-request-*.json"))
        if not requests:
            return None
        path = requests[0]
        claimed = path.with_suffix(".processing")
        try:
            path.replace(claimed)
        except FileNotFoundError:
            return None
        try:
            return _read_json(claimed)
        finally:
            claimed.unlink(missing_ok=True)

    def publish_explanation(self, explanation_id: str, value: dict[str, Any]) -> None:
        if not explanation_id or any(char not in "0123456789abcdef-" for char in explanation_id):
            raise ValueError("invalid explanation id")
        path = self.directory / "explanations" / f"{explanation_id}.json"
        _atomic_write_json(path, value)
        lock_path = self.directory / "explanation-slot.lock"
        try:
            if lock_path.read_text(encoding="utf-8") == explanation_id:
                lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def read_explanation(self, explanation_id: str) -> dict[str, Any] | None:
        if not explanation_id or any(char not in "0123456789abcdef-" for char in explanation_id):
            return None
        return _read_json(self.directory / "explanations" / f"{explanation_id}.json")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None

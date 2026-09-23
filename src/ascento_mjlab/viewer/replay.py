"""Viewer-only policy-frame ring buffer and failure/recovery capture artifacts."""

from __future__ import annotations

import gzip
import json
import math
import os
import re
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


@dataclass(frozen=True)
class TransitionSnapshot:
    """Telemetry aligned to the policy action that produced this transition."""

    episode_id: int
    episode_step: int
    sim_time_s: float
    reward_rate: float
    step_reward: float
    reward_terms: dict[str, float]
    tilt_rad: float
    tilt_rate_rad_s: float
    height_m: float
    contacts: dict[str, bool]
    balance_margin: float
    fallen: bool
    reset: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicyReplayBuffer:
    """Bounded viewer memory retaining the most recent simulation-time window."""

    def __init__(
        self,
        *,
        step_dt: float,
        capture_seconds: float = 10.0,
        margin_frames: int = 2,
    ) -> None:
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be a finite positive number")
        if not math.isfinite(capture_seconds) or capture_seconds <= 0:
            raise ValueError("capture_seconds must be a finite positive number")
        if margin_frames < 0:
            raise ValueError("margin_frames cannot be negative")
        self.capacity = math.ceil(capture_seconds / step_dt) + int(margin_frames)
        self._frames: deque[Any] = deque(maxlen=self.capacity)

    def append(self, frame: Any) -> None:
        self._frames.append(frame)

    def snapshot(self) -> list[Any]:
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


class EventTriggerDetector:
    """Identify falls and sustained recoveries from existing state diagnostics."""

    def __init__(
        self,
        *,
        unstable_margin: float = 0.35,
        stable_margin: float = 0.70,
        unstable_tilt_fraction: float = 0.65,
        recovery_dwell_s: float = 0.5,
    ) -> None:
        if not 0 <= unstable_margin < stable_margin <= 1:
            raise ValueError("recovery margins must satisfy 0 <= unstable < stable <= 1")
        if not 0 < unstable_tilt_fraction < 1:
            raise ValueError("unstable tilt fraction must be between zero and one")
        if recovery_dwell_s < 0:
            raise ValueError("recovery dwell cannot be negative")
        self.unstable_margin = unstable_margin
        self.stable_margin = stable_margin
        self.unstable_tilt_fraction = unstable_tilt_fraction
        self.recovery_dwell_s = recovery_dwell_s
        self._unstable_since: float | None = None
        self._stable_since: float | None = None
        self._episode_id: int | None = None
        self._unstable_sequence: int | None = None
        self._maximum_tilt_sequence: int | None = None
        self._maximum_tilt_rad = 0.0
        self.last_markers: list[dict[str, Any]] = []

    def update(self, frame: Any, *, fall_boundary_rad: float) -> str | None:
        transition: TransitionSnapshot = frame.transition
        self.last_markers = []
        sequence = int(frame.sequence_id)
        if transition.fallen:
            self.last_markers = [{
                "sequence": sequence,
                "kind": "fall",
                "label": "real fallen termination term",
            }]
            self._reset_episode()
            return "fall"
        if transition.reset:
            self._reset_episode()
            self._episode_id = transition.episode_id
            return None
        if self._episode_id is not None and transition.episode_id != self._episode_id:
            self._reset_episode()
        self._episode_id = transition.episode_id

        unstable = (
            transition.balance_margin < self.unstable_margin
            or transition.tilt_rad >= fall_boundary_rad * self.unstable_tilt_fraction
        )
        if self._unstable_since is None:
            if unstable:
                self._unstable_since = transition.sim_time_s
                self._unstable_sequence = sequence
                self._maximum_tilt_sequence = sequence
                self._maximum_tilt_rad = transition.tilt_rad
            return None
        if unstable:
            self._stable_since = None
            if transition.tilt_rad > self._maximum_tilt_rad:
                self._maximum_tilt_rad = transition.tilt_rad
                self._maximum_tilt_sequence = sequence
            return None
        if transition.balance_margin < self.stable_margin:
            self._stable_since = None
            return None
        if self._stable_since is None:
            self._stable_since = transition.sim_time_s
        if transition.sim_time_s - self._stable_since + 1e-9 >= self.recovery_dwell_s:
            self.last_markers = [
                {
                    "sequence": self._unstable_sequence,
                    "kind": "instability_entered",
                    "label": "unstable diagnostic threshold crossed",
                },
                {
                    "sequence": self._maximum_tilt_sequence,
                    "kind": "maximum_tilt",
                    "label": f"maximum observed tilt {self._maximum_tilt_rad:.4f} rad",
                },
                {
                    "sequence": sequence,
                    "kind": "recovery",
                    "label": "stable margin held above threshold for dwell time",
                },
            ]
            self._reset_episode()
            return "recovery"
        return None

    def _reset_episode(self) -> None:
        self._unstable_since = None
        self._stable_since = None
        self._episode_id = None
        self._unstable_sequence = None
        self._maximum_tilt_sequence = None
        self._maximum_tilt_rad = 0.0

    def reset(self) -> None:
        self._reset_episode()


class PolicyReplayRecorder:
    """Persist versioned captures as atomically replaced gzip JSON artifacts."""

    def __init__(
        self,
        directory: Path,
        *,
        viewer_id: str,
        run_id: str,
        max_events: int = 200,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.directory = Path(directory).expanduser().resolve()
        self.viewer_id = str(viewer_id)
        self.run_id = str(run_id)
        self.max_events = int(max_events)

    def persist(
        self,
        *,
        event_type: str,
        frames: Iterable[Any],
        trigger_sequence: int,
        trigger_reason: str,
        checkpoint: str,
        checkpoint_iteration: int | None,
        schema: dict[str, Any],
        policy_generation: int = 1,
        markers: list[dict[str, Any]] | None = None,
    ) -> Path:
        if event_type not in {"fall", "recovery", "manual"}:
            raise ValueError("event_type must be fall, recovery, or manual")
        frame_rows = [frame.to_dict() for frame in frames]
        if not frame_rows:
            raise ValueError("cannot persist an empty policy capture")
        created = datetime.now(timezone.utc)
        event_id = uuid4().hex
        payload = {
            "version": 1,
            "event_id": event_id,
            "event_type": event_type,
            "viewer_id": self.viewer_id,
            "run_id": self.run_id,
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": checkpoint_iteration,
            "policy_generation": int(policy_generation),
            "trigger_sequence": int(trigger_sequence),
            "trigger_reason": str(trigger_reason),
            "created_at": created.isoformat(),
            "schema": schema,
            "frames": frame_rows,
            "markers": markers or [
                {
                    "sequence": int(trigger_sequence),
                    "kind": event_type,
                    "label": str(trigger_reason),
                }
            ],
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = created.strftime("%Y%m%dT%H%M%S_%fZ")
        path = self.directory / f"event_{stamp}_{event_type}_{event_id[:8]}.json.gz"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(encoded)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(path)
        summary = {
            key: payload[key]
            for key in (
                "version",
                "event_id",
                "event_type",
                "viewer_id",
                "run_id",
                "checkpoint",
                "checkpoint_iteration",
                "policy_generation",
                "trigger_sequence",
                "trigger_reason",
                "created_at",
                "markers",
            )
        }
        summary["frame_count"] = len(frame_rows)
        summary["artifact"] = path.name
        _atomic_json(path.with_name(path.name.replace(".json.gz", ".summary.json")), summary)
        self._enforce_retention()
        return path

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("event_*.json.gz"), reverse=True)[: max(1, limit)]:
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    value = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                value["artifact"] = path.name
                rows.append(value)
        return rows

    def list_summaries(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("event_*.summary.json"), reverse=True)[: max(1, limit)]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def read_event(self, event_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[0-9a-f]{8,32}", event_id):
            return None
        for path in self.directory.glob(f"event_*_{event_id[:8]}.json.gz"):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    value = json.load(handle)
            except (OSError, json.JSONDecodeError):
                return None
            if isinstance(value, dict) and value.get("event_id") == event_id:
                value["artifact"] = path.name
                return value
        return None

    def _enforce_retention(self) -> None:
        events = sorted(
            self.directory.glob("event_*.json.gz"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in events[self.max_events :]:
            stale.unlink(missing_ok=True)
            stale.with_name(stale.name.replace(".json.gz", ".summary.json")).unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def checkpoint_iteration(checkpoint: str) -> int | None:
    match = re.search(r"model_(\d+)(?:\.pt)?$", Path(checkpoint).name)
    return int(match.group(1)) if match else None

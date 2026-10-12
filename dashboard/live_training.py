"""Read-only training telemetry discovery, independent of the policy viewer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


def find_live_frame_file(run_path: Path) -> Path | None:
    """Locate a trainer's nested log folder without traversing arbitrary trees."""
    candidates = [run_path / "live_monitor.json", *run_path.glob("*/*/live_monitor.json")]
    files = [
        candidate for candidate in candidates if candidate.is_file() and not candidate.is_symlink()
    ]
    if not files:
        return None
    try:
        return max(files, key=lambda candidate: candidate.stat().st_mtime_ns)
    except OSError:
        return None


def read_live_frame(run_path: Path, *, now: float | None = None) -> dict[str, Any]:
    """Return a validated, bounded snapshot or a clear not-yet-available result."""
    path = find_live_frame_file(run_path)
    if path is None:
        return {"available": False, "reason": "No sampled PPO telemetry from this run"}
    try:
        if path.stat().st_size > 65536:
            return {"available": False, "reason": "Telemetry payload is oversized"}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "reason": "Telemetry snapshot could not be read"}
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("source") != "actual_ppo_rollout"
        or not isinstance(data.get("environments"), list)
        or len(data["environments"]) > 4
        or not isinstance(data.get("sequence"), int)
    ):
        return {"available": False, "reason": "Invalid sampled telemetry schema"}
    captured = data.get("sampled_at")
    if not isinstance(captured, (float, int)) or not math.isfinite(captured):
        return {"available": False, "reason": "Invalid timestamp"}
    age = max(0.0, (time.time() if now is None else now) - captured)
    return {**data, "available": True, "stale": age > 10.0, "age_seconds": round(age, 2)}

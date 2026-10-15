"""Read-only summaries for fresh PPO streams across isolated worktrees."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dashboard import deep_live

_ALIAS = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
_GPU_LOCK = threading.Lock()
_GPU_CACHE: tuple[float, dict | None] = (0.0, None)


def metadata_roots(raw: str | None = None) -> dict[str, Path]:
    """Only operator-configured, separately mounted read-only folders may be read."""
    value = os.environ.get("ASCENTO_DEEP_LIVE_METADATA_ROOTS", "") if raw is None else raw
    roots: dict[str, Path] = {}
    for part in filter(None, (x.strip() for x in value.split(";"))):
        alias, separator, location = part.partition("=")
        if not separator or not _ALIAS.fullmatch(alias) or alias in roots:
            raise ValueError("Invalid live metadata alias")
        path = Path(location)
        if not path.is_absolute():
            raise ValueError("Live metadata roots must be absolute")
        roots[alias] = path.resolve()
    return roots


def _read_status(root: Path) -> dict:
    path = root / "status.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        updated = datetime.fromisoformat(payload["updated_at"])
        if updated.tzinfo is None or abs((datetime.now(timezone.utc) - updated).total_seconds()) > 120:
            return {}
        return payload
    except (OSError, ValueError, TypeError, KeyError, UnicodeDecodeError):
        return {}


def _tail_metrics(root: Path) -> tuple[list[dict], int | None]:
    path = root / "trainer.log"
    try:
        if path.is_symlink() or not path.is_file() or time.time() - path.stat().st_mtime > 120:
            return [], None
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 3 * 1024 * 1024))
            lines = handle.read(3 * 1024 * 1024).decode("utf-8", errors="replace").splitlines()
    except OSError:
        return [], None

    iteration: int | None = None
    samples: list[dict] = []
    stage: int | None = None
    for line in lines:
        match = _ITERATION.search(line)
        if match:
            iteration = int(match.group(1))
        elif iteration is not None and "Mean reward:" in line:
            try:
                reward = float(line.split("Mean reward:", 1)[1].strip())
                if math.isfinite(reward):
                    samples.append({"iteration": iteration, "reward": reward})
            except ValueError:
                pass
        elif "Episode_Metrics/generalist_curriculum_goal_mix_stage:" in line:
            try:
                value = float(line.rsplit(":", 1)[1])
                if math.isfinite(value) and value in (0, 1, 2):
                    stage = int(value)
            except ValueError:
                pass
    return list({v["iteration"]: v for v in samples}.values())[-80:], stage


def _remaining(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        end = datetime.fromisoformat(value)
        if end.tzinfo is None:
            return None
        return max(0, int((end - datetime.now(timezone.utc)).total_seconds()))
    except ValueError:
        return None


def _gpu() -> dict | None:
    """Cached nvidia-smi query; read-only and does not create a CUDA context."""
    global _GPU_CACHE
    with _GPU_LOCK:
        stamp, cached = _GPU_CACHE
        if time.monotonic() - stamp < 12:
            return cached
        result = None
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=2, check=True,
            )
            values = [float(x.strip()) for x in proc.stdout.splitlines()[0].split(",")]
            if len(values) == 4 and all(math.isfinite(x) and x >= 0 for x in values):
                result = dict(zip(
                    ("utilization_percent", "memory_used_mib", "memory_total_mib", "temperature_c"),
                    values,
                    strict=True,
                ))
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, IndexError):
            pass
        _GPU_CACHE = (time.monotonic(), result)
        return result


def summary(primary: Path) -> dict:
    live = deep_live.discover_live_sources(primary)["live_runs"]
    roots = metadata_roots()
    per_root = Counter(source["root"] for source in live)
    rows = []
    for source in live:
        # A supervisor log is per worktree, not per live source. If multiple
        # policies stream from one root, do not attribute its metrics arbitrarily.
        folder = roots.get(source["root"]) if per_root[source["root"]] == 1 else None
        status = _read_status(folder) if folder and folder.is_dir() else {}
        trend, stage = _tail_metrics(folder) if folder and folder.is_dir() else ([], None)
        reward = status.get("last_mean_reward")
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            reward = None
        rows.append({
            "id": source["id"],
            "root": source["root"],
            "label": source["label"],
            "state": source["state"],
            "sampled_at": source["sampled_at"],
            "iteration": status.get("last_iteration") if isinstance(status.get("last_iteration"), int) else None,
            "mean_reward": reward,
            "reward_trend": trend,
            "curriculum_stage": stage,
            "curriculum_stage_count": 3 if stage is not None else None,
            "training_remaining_seconds": _remaining(status.get("training_cutoff_local")),
            "deadline_remaining_seconds": _remaining(status.get("deadline_local")),
            "training_cutoff": status.get("training_cutoff_local"),
            "deadline": status.get("deadline_local"),
            "metadata_available": bool(status),
        })
    return {"runs": rows, "gpu": _gpu() if rows else None}

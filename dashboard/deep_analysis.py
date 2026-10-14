"""Read-only, checkpoint-scoped deep analysis; never simulates or evaluates policies."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/deep-analysis", tags=["deep-analysis"])
_CACHE_LOCK = threading.Lock()
_CATALOG_CACHE: tuple[float, Path, dict] | None = None
_MAX_CAPTURE_SIZE = 64 * 1024 * 1024
_MAX_SAMPLES = 600
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


def _inside(path: Path, root: Path) -> bool:
    """Symlink-safe: reject all symlink components, not just the final file."""
    try:
        path.relative_to(root)
        return path.resolve().is_relative_to(root.resolve()) and not any(
            part.is_symlink() for part in [path, *path.parents] if part != root
        )
    except (ValueError, OSError):
        return False


def _checkpoints(run_root: Path) -> list[dict]:
    entries = []
    if not run_root.is_dir():
        return entries
    for candidate in run_root.glob("**/model_*.pt"):
        try:
            if not candidate.is_file() or not _inside(candidate, run_root):
                continue
            matched = _CHECKPOINT_PATTERN.fullmatch(candidate.name)
            if not matched:
                continue
            relative = candidate.relative_to(run_root).as_posix()
            ancestor = candidate.relative_to(run_root).parts[0]
            st = candidate.stat()
            entries.append(
                {
                    "id": relative,
                    "run": ancestor,
                    "run_label": ancestor,
                    "iteration": int(matched.group(1)),
                    "modified_at": st.st_mtime,
                    "size_bytes": st.st_size,
                    "path": relative,
                    "lineage": ancestor,
                }
            )
        except (ValueError, OSError):
            continue
    entries.sort(key=lambda item: (item["modified_at"], item["iteration"]), reverse=True)
    return entries


def _checkpoint_key(value: str, run_root: Path) -> str | None:
    """Map checkpoint provenance to a run-relative ID without trusting a filename.

    The same canonical checkout is mounted at /workspace in Docker but captures
    recorded outside Docker retain their absolute host path. Only two explicit
    trusted roots are allowed: the actual runtime artifact root and the canonical
    source root configured by the deployment.
    """
    raw = Path(value)
    if not raw.is_absolute():
        return raw.as_posix() if raw.parts and ".." not in raw.parts else None
    roots = [run_root.resolve()]
    canonical = os.environ.get("ASCENTO_CANONICAL_SOURCE_ROOT")
    if canonical:
        roots.append(Path(canonical) / "logs" / "rsl_rl")
    for root in roots:
        try:
            relative = raw.relative_to(root)
            if relative.parts and ".." not in relative.parts:
                return relative.as_posix()
        except ValueError:
            continue
    return None


def _capture_index(capture_root: Path, run_root: Path, allowed: set[str]) -> dict[str, list[dict]]:
    recordings: dict[str, list[dict]] = {}
    if not capture_root.is_dir():
        return recordings
    for path in capture_root.rglob("*.npz"):
        try:
            if (
                not path.is_file()
                or not _inside(path, capture_root)
                or path.stat().st_size > _MAX_CAPTURE_SIZE
            ):
                continue
            with np.load(path, allow_pickle=False) as data:
                if "meta_checkpoint" not in data:
                    continue
                checkpoint_value = str(data["meta_checkpoint"].item())
                key = _checkpoint_key(checkpoint_value, run_root)
                if key not in allowed:
                    continue
                scenario = (
                    str(data["meta_scenario_id"].item()) if "meta_scenario_id" in data else None
                )
                task = str(data["meta_task"].item()) if "meta_task" in data else None
                relative = path.relative_to(capture_root).as_posix()
                rrd = path.with_suffix(".rrd")
                recordings.setdefault(key, []).append(
                    {
                        "id": relative,
                        "scenario": scenario,
                        "task": task,
                        "frames": int(len(data["time"])) if "time" in data else 0,
                        "rrd_url": (
                            "/api/replays/files/"
                            + quote(rrd.relative_to(capture_root).as_posix(), safe="/")
                            if rrd.is_file() and _inside(rrd, capture_root)
                            else None
                        ),
                        "updated_at": path.stat().st_mtime,
                    }
                )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for values in recordings.values():
        values.sort(key=lambda item: item["updated_at"], reverse=True)
    return recordings


def catalog(run_root: Path, capture_root: Path, active_runs: list[dict] | None = None) -> dict:
    """Identity-aware inventory. Live run identities are passed by the dashboard service."""
    global _CATALOG_CACHE
    run_root = run_root.resolve()
    capture_root = capture_root.resolve()
    with _CACHE_LOCK:
        cached = _CATALOG_CACHE
    if cached is not None and cached[1] == run_root and time.monotonic() - cached[0] < 20:
        base = cached[2]
    else:
        checkpoints = _checkpoints(run_root)
        recordings = _capture_index(capture_root, run_root, {item["id"] for item in checkpoints})
        for item in checkpoints:
            item["recording_count"] = len(recordings.get(item["id"], []))
        base = {"checkpoints": checkpoints, "recordings": recordings}
        with _CACHE_LOCK:
            _CATALOG_CACHE = (time.monotonic(), run_root, base)
    return {
        "checkpoints": base["checkpoints"],
        "recordings": base["recordings"],
        "live_runs": [
            {
                "id": str(row["id"]),
                "label": str(row.get("display_name") or row.get("name") or row["id"]),
                "state": str(row.get("state")),
                "iteration": row.get("iteration"),
                "stale": bool(row.get("stale", False)),
            }
            for row in (active_runs or [])
            if row.get("state") in ("running", "starting", "stopping")
            and not row.get("stale", False)
        ],
        "note": "Live uses in-memory PPO observations, not a frozen checkpoint or benchmark evaluation.",
    }


def _require_recording(capture_root: Path, relative: str) -> Path:
    root = capture_root.resolve()
    if not relative or "\x00" in relative or Path(relative).is_absolute():
        raise HTTPException(status_code=404, detail="Recording not found")
    path = root / relative
    if path.suffix != ".npz" or not path.is_file() or not _inside(path, root):
        raise HTTPException(status_code=404, detail="Recording not found")
    if path.stat().st_size > _MAX_CAPTURE_SIZE:
        raise HTTPException(status_code=413, detail="Capture too large for interactive analysis")
    return path


def _finite(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _sample(data: object, name: str, positions: np.ndarray, n: int, width: int | None = None):
    if name not in data:
        return None
    array = np.asarray(data[name])
    if array.shape != ((n,) if width is None else (n, width)):
        return None
    values = array[positions].astype(float)
    if width is None:
        return [_finite(v) for v in values]
    return [[_finite(v) for v in row] for row in values]


def _stats(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": _finite(values.mean()),
        "p50": _finite(np.median(values)),
        "p95": _finite(np.quantile(values, 0.95)),
        "max": _finite(values.max()),
        "min": _finite(values.min()),
    }


def episode(capture_root: Path, run_root: Path, checkpoint_id: str, recording_id: str) -> dict:
    """Read-only, bounded payload. The checkpoint must match recorded provenance."""
    valid = {entry["id"] for entry in _checkpoints(run_root)}
    if checkpoint_id not in valid:
        raise HTTPException(404, "Checkpoint not found")
    capture = _require_recording(capture_root, recording_id)
    try:
        with np.load(capture, allow_pickle=False) as d:
            if "meta_checkpoint" not in d:
                raise HTTPException(409, "Capture has no checkpoint provenance")
            actual = _checkpoint_key(str(d["meta_checkpoint"].item()), run_root)
            if actual != checkpoint_id:
                raise HTTPException(409, "Recording belongs to another checkpoint")
            timestamp = np.asarray(d["time"], dtype=float)
            n = len(timestamp)
            if not n or not np.isfinite(timestamp).all() or (np.diff(timestamp) <= 0).any():
                raise HTTPException(422, "Invalid capture timeline")
            indices = np.unique(np.r_[np.arange(0, n, max(1, math.ceil(n / _MAX_SAMPLES))), n - 1])
            scalar_names = (
                "tilt_rad",
                "target_error_m",
                "heading_error_rad",
                "planar_speed_mps",
                "angular_xy_radps",
                "yaw_rate_radps",
                "reward_total",
                "evaluator_stop_active",
                "evaluator_stop_all_pass",
                "evaluator_stop_dwell_s",
                "evaluator_recovery_stable_dwell_s",
                "evaluator_recovered",
            )
            scalar_names += tuple(
                k
                for k in d.keys()
                if k.startswith("evaluator_stop_") or k.startswith("evaluator_recovery_")
            )
            channels = {k: _sample(d, k, indices, n) for k in dict.fromkeys(scalar_names)}
            channels = {k: v for k, v in channels.items() if v is not None}
            for name, width in (
                ("root_pos", 3),
                ("target_xy", 2),
                ("contacts", 2),
                ("root_lin_vel_w", 3),
                ("root_ang_vel_b", 3),
                ("action", 6),
                ("effort", 6),
                ("joint_pos", 6),
                ("joint_vel", 6),
            ):
                values = _sample(d, name, indices, n, width)
                if values is not None:
                    channels[name] = values
            # Reward rates as logged. Convert to step contribution using reward_dt.
            rewards = {}
            if all(k in d for k in ("reward_terms_rate", "reward_dt", "meta_reward_term_names")):
                labels = json.loads(str(d["meta_reward_term_names"].item()))
                rates = np.asarray(d["reward_terms_rate"])
                dt = np.asarray(d["reward_dt"])
                if rates.shape == (n, len(labels)) and dt.shape == (n,):
                    for index, label in enumerate(labels):
                        contribution = rates[indices, index] * dt[indices]
                        rewards[f"{index:02d} {label}"] = [_finite(x) for x in contribution]
            stats = {}
            for name in (
                "tilt_rad",
                "target_error_m",
                "planar_speed_mps",
                "heading_error_rad",
                "reward_total",
            ):
                if name in d and np.asarray(d[name]).shape == (n,):
                    stats[name] = _stats(np.asarray(d[name], dtype=float))
                    stats[name + "_final_2s"] = _stats(
                        np.asarray(d[name], dtype=float)[timestamp >= timestamp[-1] - 2]
                    )
            for name in (
                "evaluator_stop_all_pass",
                "evaluator_stop_tilt",
                "evaluator_stop_distance",
                "evaluator_stop_both_wheels",
                "evaluator_recovered",
            ):
                if name in d:
                    vals = np.asarray(d[name], dtype=float)
                    stats[name + "_final_2s"] = _stats(vals[timestamp >= timestamp[-1] - 2])
            return {
                "checkpoint_id": checkpoint_id,
                "recording_id": recording_id,
                "scenario": str(d["meta_scenario_id"].item()) if "meta_scenario_id" in d else None,
                "task": str(d["meta_task"].item()) if "meta_task" in d else None,
                "checkpoint_sha256": str(d["meta_checkpoint_sha256"].item())
                if "meta_checkpoint_sha256" in d
                else None,
                "suite_sha256": str(d["meta_suite_sha256"].item())
                if "meta_suite_sha256" in d
                else None,
                "frames": n,
                "duration_s": _finite(timestamp[-1] - timestamp[0]),
                "sampled_frames": len(indices),
                "time": [_finite(x) for x in timestamp[indices]],
                "channels": channels,
                "reward_terms": rewards,
                "stats": stats,
                "has_evaluator_signals": any(k.startswith("evaluator_") for k in d.keys()),
                "coverage_note": "Exploratory replay measurements; recorded evaluator flags are observations, not a new evaluation.",
            }
    except HTTPException:
        raise
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(422, f"Invalid recording: {exc}") from exc

"""FastAPI service for remotely monitoring and managing Ascento PPO training."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ascento_mjlab.viewer.ipc import CaptureBusyError, ExplanationBusyError
from dashboard.assessments import build_assessments
from dashboard.checkpoint_registry import checkpoint_evidence
from dashboard.config import load_config, validate_startup
from dashboard.control_auth import (
    CONTROL_COOKIE_NAME,
    configured_token,
    issue_session,
    session_remaining_seconds,
    token_matches,
    valid_session,
)
from dashboard.curriculum import curriculum_for_run
from dashboard.database import DashboardDatabase
from dashboard.evaluations import (
    compare_evaluations,
    discover_evaluations,
    discover_suites,
    evaluation_detail,
    evaluation_scenarios,
)
from dashboard.experiments import discover_experiments
from dashboard.health import (
    build_run_info,
    decorate_records,
    discover_dashboard_runs,
    gpu_snapshot,
    list_dashboard_summaries,
    load_dashboard_records,
    process_status,
)
from dashboard.monitor import load_training_records, tail_lines, training_log_path
from dashboard.run_service import RunService
from dashboard.runtime_policy import RuntimePolicyError, runtime_identity
from dashboard.supervisor_client import (
    SupervisorClient,
    SupervisorRejected,
    SupervisorUnavailable,
)
from dashboard.task_catalog import task_catalog, task_ids
from dashboard.versioning import current_repository_version, runtime_revision_report
from dashboard.viewer_service import ViewerBusyError, ViewerNotFoundError, ViewerService

CONFIG = load_config()
STARTUP_WARNINGS = validate_startup(CONFIG, create_artifact_root=False)
ARTIFACT_ROOT = CONFIG.artifact_root
FRONTEND_DIST = CONFIG.frontend_dist
BLENDER_RENDER_ROOT = CONFIG.repo_root / "captures" / "blender"
EVALUATION_ROOT = CONFIG.repo_root / "evaluations"
EVALUATION_SUITE_ROOT = CONFIG.repo_root / "benchmarks" / "suites"
EXPERIMENT_ROOT = CONFIG.repo_root / "docs" / "experiments"
BLENDER_RENDER_ASSET_SUFFIXES = {".blend", ".mp4", ".png"}
RUN_SERVICE = RunService(ARTIFACT_ROOT, stale_after_seconds=CONFIG.stale_after_seconds)
VIEWER_SERVICE = ViewerService(
    RUN_SERVICE,
    logs_root=CONFIG.repo_root / "logs" / "viewers",
    host=os.environ.get("ASCENTO_VIEWER_HOST", "0.0.0.0"),
    port=int(os.environ.get("ASCENTO_VIEWER_PORT", "8081")),
    stable_age_seconds=float(os.environ.get("ASCENTO_VIEWER_STABLE_AGE_SECONDS", "2.0")),
)
SUPERVISOR = SupervisorClient()
DATABASE = DashboardDatabase(CONFIG.database_url, CONFIG.repo_root)

# Artifact discovery walks a mounted training directory.  Keeping the annotated
# list briefly avoids making every UI poll repeat that full filesystem scan.
_SUMMARY_CACHE_TTL_S = 25.0
_SUMMARY_CACHE_LOCK = threading.Lock()
_SUMMARY_CACHE: tuple[float, list[dict]] | None = None
_INDEX_CACHE_TTL_S = 10.0
_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_CACHE: tuple[float, list[dict]] | None = None
_OVERVIEW_SERIES_CACHE_TTL_S = 10.0
_OVERVIEW_SERIES_CACHE_LOCK = threading.Lock()
_OVERVIEW_SERIES_CACHE: dict[str, tuple[float, list[dict]]] = {}
_EVALUATION_CACHE_TTL_S = 10.0
_EVALUATION_CACHE_LOCK = threading.Lock()
_EVALUATION_CACHE: tuple[float, list[dict]] | None = None

app = FastAPI(title="Ascento Control", version="3.0")


class RunCreateRequest(BaseModel):
    display_name: str
    task: str = "Ascento-Balance-Flat"
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    purpose: str = ""
    experiment_id: str | None = None
    parent_run_id: str | None = None
    parent_checkpoint: str | None = None
    episode_horizon_s: float | None = Field(default=None)
    max_speed_mps: float | None = Field(default=None)
    allow_dirty_provenance: bool = False
    training_args: list[str] = Field(default_factory=list)


class ControlSessionRequest(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class RunUpdateRequest(BaseModel):
    display_name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    purpose: str | None = None
    parent_run_id: str | None = None
    parent_checkpoint: str | None = None


class RunStopRequest(BaseModel):
    reason: str = "user_requested"


class ViewerCreateRequest(BaseModel):
    run_id: str
    checkpoint: str = "latest"
    follow: bool = False
    device: str | None = None
    jacobian_hz: float = 2.0


class ViewerWaypointRequest(BaseModel):
    operation: Literal["set", "queue", "hold", "resume", "cancel", "speed"]
    x_m: float | None = None
    y_m: float | None = None
    yaw_rad: float | None = None
    speed_mps: float | None = None


class ViewerExplanationRequest(BaseModel):
    checkpoint: str
    input_raw: list[float]
    action_index: int
    action_name: str
    baseline_kind: Literal["normalizer_mean", "episode_start", "selected_frame"]
    baseline_raw: list[float] | None = None
    baseline_description: str = ""
    n_steps: int = 32
    live_sequence: int | None = None
    episode_id: int | None = None
    event_id: str | None = None


def _require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        raise HTTPException(status_code=403, detail="control requests require a same-origin Origin header")
    if request.headers.get("sec-fetch-site", "same-origin").lower() == "cross-site":
        raise HTTPException(status_code=403, detail="cross-site control requests are not allowed")
    try:
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("invalid Origin")
        normalized_origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    except ValueError as error:
        raise HTTPException(status_code=403, detail="invalid control request Origin") from error
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    scheme = forwarded_proto if forwarded_proto in {"http", "https"} else request.url.scheme.lower()
    host = request.headers.get("host", "").lower()
    allowed = {
        f"{scheme}://{host}".rstrip("/"),
        *(
            value.strip().rstrip("/").lower()
            for value in os.environ.get("ASCENTO_DASHBOARD_ALLOWED_ORIGINS", "").split(",")
            if value.strip()
        ),
    }
    if normalized_origin not in allowed:
        raise HTTPException(status_code=403, detail="control request Origin does not match this dashboard")


def _require_control_session(request: Request) -> None:
    _require_same_origin(request)
    if not valid_session(configured_token(), request.cookies.get(CONTROL_COOKIE_NAME)):
        raise HTTPException(status_code=403, detail="unlock dashboard controls with the configured control token")


@app.get("/api/control/session")
def control_session_status(request: Request):
    token = configured_token()
    expires_in = session_remaining_seconds(token, request.cookies.get(CONTROL_COOKIE_NAME))
    authenticated = expires_in is not None
    return JSONResponse({
        "configured": token is not None,
        "authenticated": authenticated,
        "expires_in_seconds": expires_in,
    }, headers={"Cache-Control": "no-store"})


@app.post("/api/control/session")
def open_control_session(payload: ControlSessionRequest, request: Request):
    _require_same_origin(request)
    token = configured_token()
    if token is None:
        raise HTTPException(
            status_code=503,
            detail="set ASCENTO_CONTROL_TOKEN to a secret of at least 32 characters before enabling dashboard controls",
        )
    if not token_matches(token, payload.token):
        raise HTTPException(status_code=403, detail="control token was not accepted")
    response = JSONResponse({"authenticated": True, "expires_in_seconds": 12 * 60 * 60})
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    response.set_cookie(
        CONTROL_COOKIE_NAME,
        issue_session(token),
        max_age=12 * 60 * 60,
        httponly=True,
        secure=request.url.scheme == "https" or forwarded_proto == "https",
        samesite="strict",
        path="/",
    )
    return response


@app.delete("/api/control/session")
def close_control_session(request: Request):
    _require_same_origin(request)
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(CONTROL_COOKIE_NAME, path="/", httponly=True, samesite="strict")
    return response


def _run(run_id: str):
    try:
        return RUN_SERVICE.resolve(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"failed to scan artifact root {ARTIFACT_ROOT}: {error}",
        ) from error


def _sample_records(records: list[dict], max_points: int) -> list[dict]:
    """Keep the complete run span while bounding the payload sent to the browser."""
    if len(records) <= max_points:
        return records
    last_index = len(records) - 1
    return [records[round(index * last_index / (max_points - 1))] for index in range(max_points)]


def _telemetry_coverage(records: list[dict]) -> dict[str, dict[str, int]]:
    """Describe which canonical series are genuinely available after sampling."""
    keys = sorted(
        {str(key) for record in records for key in (record.get("canonical_metrics") or {})}
    )
    return {
        key: {
            "present": sum(
                1
                for record in records
                if isinstance((record.get("canonical_metrics") or {}).get(key), (int, float))
                and not isinstance((record.get("canonical_metrics") or {}).get(key), bool)
                and math.isfinite(float((record.get("canonical_metrics") or {})[key]))
            ),
            "missing": sum(
                1
                for record in records
                if not (
                    isinstance((record.get("canonical_metrics") or {}).get(key), (int, float))
                    and not isinstance((record.get("canonical_metrics") or {}).get(key), bool)
                    and math.isfinite(float((record.get("canonical_metrics") or {})[key]))
                )
            ),
        }
        for key in keys
    }


def _artifact_health() -> list[str]:
    problems: list[str] = []
    if ARTIFACT_ROOT.exists() and not ARTIFACT_ROOT.is_dir():
        problems.append(f"artifact root is not a directory: {ARTIFACT_ROOT}")
    elif ARTIFACT_ROOT.exists():
        if not os.access(ARTIFACT_ROOT, os.R_OK):
            problems.append(f"artifact root is not readable: {ARTIFACT_ROOT}")
        if not os.access(ARTIFACT_ROOT, os.X_OK):
            problems.append(f"artifact root is not searchable: {ARTIFACT_ROOT}")
    return problems


def _blender_asset_url(path: Path, root: Path) -> str | None:
    try:
        resolved_root = root.resolve()
        resolved_path = path.resolve(strict=True)
        relative_path = resolved_path.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if not resolved_path.is_file():
        return None
    if (
        resolved_path.suffix.lower() not in BLENDER_RENDER_ASSET_SUFFIXES
        and not resolved_path.name.endswith(".manifest.json")
    ):
        return None
    return f"/api/blender/renders/files/{quote(relative_path.as_posix(), safe='/')}"


def _invalidate_summary_cache() -> None:
    global _SUMMARY_CACHE, _INDEX_CACHE
    with _SUMMARY_CACHE_LOCK:
        _SUMMARY_CACHE = None
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE = None


def _evaluation_summaries() -> list[dict]:
    global _EVALUATION_CACHE
    now = time.monotonic()
    with _EVALUATION_CACHE_LOCK:
        if _EVALUATION_CACHE is not None and now - _EVALUATION_CACHE[0] < _EVALUATION_CACHE_TTL_S:
            return list(_EVALUATION_CACHE[1])
    rows = discover_evaluations(EVALUATION_ROOT, limit=5000)
    with _EVALUATION_CACHE_LOCK:
        _EVALUATION_CACHE = (now, rows)
    return list(rows)


def _evaluation_etag(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return '"' + hashlib.sha256(encoded).hexdigest() + '"'


def _compact_run(summary: dict) -> dict:
    telemetry = summary.get("telemetry") if isinstance(summary.get("telemetry"), dict) else {}
    canonical = (
        telemetry.get("canonical_metrics")
        if isinstance(telemetry.get("canonical_metrics"), dict)
        else {}
    )
    status = summary.get("status") if isinstance(summary.get("status"), dict) else {}
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    repository = (
        summary.get("repository_version")
        if isinstance(summary.get("repository_version"), dict)
        else {}
    )
    return {
        "id": summary.get("id"),
        "display_name": summary.get("display_name") or summary.get("name"),
        "name": summary.get("name"),
        "task": status.get("task"),
        "stage": summary.get("stage"),
        "state": summary.get("state"),
        "stale": bool(summary.get("stale")),
        "freshness_seconds": summary.get("freshness_seconds"),
        "modified_at": summary.get("modified_at"),
        "tags": summary.get("tags") or [],
        "purpose": metadata.get("purpose") or "",
        "lineage": summary.get("lineage") or {},
        "repository_version": {
            "status": repository.get("status"),
            "is_outdated": bool(repository.get("is_outdated")),
            "run_commit": repository.get("run_commit"),
            "current_commit": repository.get("current_commit"),
        },
        "iteration": telemetry.get("iteration"),
        "total_iterations": telemetry.get("total_iterations"),
        "percent_complete": telemetry.get("percent_complete"),
        "eta_seconds": telemetry.get("eta_seconds"),
        "throughput": telemetry.get("environment_steps_per_second"),
        "reward": canonical.get("reward"),
        "episode_length": canonical.get("episode_length"),
        "kl": canonical.get("kl"),
        "entropy": canonical.get("entropy"),
        "ppo_loss": canonical.get("ppo_loss"),
        "clip_fraction": canonical.get("clip_fraction"),
        "invalid_update": canonical.get("invalid_update"),
    }


def _indexed_summaries() -> list[dict]:
    global _INDEX_CACHE
    with _INDEX_CACHE_LOCK:
        if _INDEX_CACHE is not None:
            cached_at, rows = _INDEX_CACHE
            if time.monotonic() - cached_at < _INDEX_CACHE_TTL_S:
                return list(rows)

    summaries = list_dashboard_summaries(
        ARTIFACT_ROOT,
        stale_after_seconds=CONFIG.stale_after_seconds,
    )
    refs = {ref.id: ref for ref in discover_dashboard_runs(ARTIFACT_ROOT)}
    rows: list[dict] = []
    for summary in summaries:
        ref = refs.get(summary.get("id"))
        if ref is None:
            continue
        RUN_SERVICE.annotate_index(summary, ref.path)
        row = _compact_run(summary)
        if row["state"] in {"starting", "running", "stopping"}:
            try:
                row = _compact_run(RUN_SERVICE.progress_index(str(row["id"])))
            except (KeyError, OSError):
                pass
        rows.append(row)
        DATABASE.sync_run(row)

    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE = (time.monotonic(), rows)
    return list(rows)


def _overview_series(run_id: str, max_points: int = 120) -> list[dict]:
    with _OVERVIEW_SERIES_CACHE_LOCK:
        cached = _OVERVIEW_SERIES_CACHE.get(run_id)
        if cached and time.monotonic() - cached[0] < _OVERVIEW_SERIES_CACHE_TTL_S:
            return list(cached[1])

    ref = _run(run_id)
    raw = load_training_records(ref.path, limit=600)
    if len(raw) > max_points:
        raw = _sample_records(raw, max_points)
    records = decorate_records(raw, ref.path)
    result = []
    for record in records:
        canonical = record.get("canonical_metrics") or {}
        result.append(
            {
                "iteration": record.get("iteration"),
                "reward": canonical.get("reward"),
                "episode_length": canonical.get("episode_length"),
                "kl": canonical.get("kl"),
                "entropy": canonical.get("entropy"),
                "ppo_loss": canonical.get("ppo_loss"),
                "clip_fraction": canonical.get("clip_fraction"),
            }
        )
    with _OVERVIEW_SERIES_CACHE_LOCK:
        _OVERVIEW_SERIES_CACHE[run_id] = (time.monotonic(), result)
    return list(result)


def _curriculum_snapshot(run_id: str) -> tuple[dict, dict | None]:
    """Build curriculum state without loading detailed contracts or GPU diagnostics."""
    ref = _run(run_id)
    progress = RUN_SERVICE.progress_index(run_id)
    run_info = build_run_info(
        ref.path,
        ARTIFACT_ROOT,
        str(progress.get("stage") or "unknown"),
    )
    detail = {**progress, "run_info": run_info}
    return detail, curriculum_for_run(detail)


def _overview_run(detail: dict) -> dict:
    """Project a cheap progress snapshot into the landing-page run model."""
    row = _compact_run(detail)
    run_info = detail.get("run_info") if isinstance(detail.get("run_info"), dict) else {}
    row.update(
        {
            "task": run_info.get("task") or row.get("task"),
            "started_at": run_info.get("started_at"),
            "device": run_info.get("device"),
            "invalid_updates": 1 if row.get("invalid_update") else 0,
            "non_finite_updates": 0,
        }
    )
    return row


def _cached_summary_count() -> int | None:
    with _SUMMARY_CACHE_LOCK:
        if _SUMMARY_CACHE is None:
            return None
        cached_at, summaries = _SUMMARY_CACHE
        if time.monotonic() - cached_at >= _SUMMARY_CACHE_TTL_S:
            return None
        return len(summaries)


def _annotated_summaries() -> list[dict]:
    global _SUMMARY_CACHE
    with _SUMMARY_CACHE_LOCK:
        if _SUMMARY_CACHE is not None:
            cached_at, summaries = _SUMMARY_CACHE
            if time.monotonic() - cached_at < _SUMMARY_CACHE_TTL_S:
                return list(summaries)

    # Keep the lock out of the filesystem walk so a health request is never
    # held behind a slow network or mounted-drive scan.
    summaries = list_dashboard_summaries(
        ARTIFACT_ROOT,
        stale_after_seconds=CONFIG.stale_after_seconds,
    )
    refs = {ref.id: ref for ref in discover_dashboard_runs(ARTIFACT_ROOT)}
    for summary in summaries:
        ref = refs.get(summary.get("id"))
        if ref is not None:
            RUN_SERVICE.annotate(summary, ref.path)

    with _SUMMARY_CACHE_LOCK:
        _SUMMARY_CACHE = (time.monotonic(), summaries)
    return list(summaries)


@app.get("/api/health")
def health():
    return _health_snapshot()


def _component(status: str, *, message: str | None = None, checked_at: float | None = None) -> dict:
    value = {"status": status}
    if message:
        value["message"] = message
    if checked_at is not None:
        value["checked_at"] = checked_at
    return value


def _system_components(
    *, checked_at: float | None = None, database_status: dict | None = None
) -> dict[str, dict]:
    checked_at = time.time() if checked_at is None else checked_at
    problems = _artifact_health()
    database = database_status if database_status is not None else DATABASE.status()
    if not database.get("enabled"):
        database_component = _component("optional", message="filesystem index is active", checked_at=checked_at)
    elif database.get("available"):
        database_component = _component("healthy", checked_at=checked_at)
    else:
        database_component = _component(
            "degraded", message=str(database.get("error") or "database unavailable"), checked_at=checked_at
        )

    try:
        supervisor = SUPERVISOR.status()
    except SupervisorUnavailable as error:
        supervisor_component = _component("unavailable", message=str(error), checked_at=checked_at)
        tailscale = _component(
            "unknown", message="cannot verify without the host supervisor", checked_at=checked_at
        )
    else:
        supervisor_component = _component("healthy", checked_at=checked_at)
        tailscale_state = supervisor.get("tailscale")
        if isinstance(tailscale_state, dict) and tailscale_state.get("connected") is True:
            tailscale = _component("healthy", checked_at=checked_at)
        elif isinstance(tailscale_state, dict) and tailscale_state.get("connected") is False:
            tailscale = _component("degraded", message="Tailscale is disconnected", checked_at=checked_at)
        else:
            tailscale = _component("unknown", message="supervisor returned no Tailscale state", checked_at=checked_at)

    frontend_index = FRONTEND_DIST / "index.html"
    frontend = (
        _component("healthy", checked_at=checked_at)
        if frontend_index.is_file()
        else _component("unavailable", message="frontend build is missing", checked_at=checked_at)
    )
    artifact_status = "unavailable" if problems else "healthy"
    artifact_message = "; ".join(problems) if problems else None
    return {
        "api": _component("healthy", checked_at=checked_at),
        "frontend": frontend,
        "database": database_component,
        "supervisor": supervisor_component,
        "tailscale": tailscale,
        "artifacts": _component(artifact_status, message=artifact_message, checked_at=checked_at),
    }


def _health_snapshot() -> dict:
    checked_at = time.time()
    problems = _artifact_health()
    database = DATABASE.status()
    components = _system_components(checked_at=checked_at, database_status=database)
    # A missing artifact directory is an expected empty state for a read-only
    # dashboard. Readiness fails only when the configured root is unusable.
    ready = not problems
    degraded = any(
        component["status"] in {"degraded", "unavailable"}
        for name, component in components.items()
        if name not in {"api", "frontend", "artifacts"} or component["status"] == "unavailable"
    )
    status = "unavailable" if not ready else "degraded" if degraded else "healthy"
    # Health is polled alongside /api/runs.  Do not make it perform a second
    # full artifact traversal just to produce an informational count.
    run_count = _cached_summary_count()
    if run_count is None and not ARTIFACT_ROOT.exists():
        run_count = 0
    return {
        "ok": status == "healthy",
        "live": True,
        "ready": ready,
        "status": status,
        "checked_at": checked_at,
        "artifact_root": str(ARTIFACT_ROOT),
        "run_count": run_count,
        "problems": problems,
        "components": components,
        "warnings": STARTUP_WARNINGS,
        "config": CONFIG.public_dict(),
        "repository_version": current_repository_version(),
        "database": database,
    }


@app.get("/api/health/live")
def health_live():
    return {"live": True, "status": "healthy", "checked_at": time.time()}


@app.get("/api/health/ready")
def health_ready():
    snapshot = _health_snapshot()
    if not snapshot["ready"]:
        return JSONResponse(status_code=503, content=snapshot)
    return {"ready": True, "status": snapshot["status"], "checked_at": snapshot["checked_at"], "problems": []}


@app.get("/api/system/components")
def system_components():
    checked_at = time.time()
    database = DATABASE.status()
    return {
        "checked_at": checked_at,
        "components": _system_components(checked_at=checked_at, database_status=database),
    }


@app.get("/api/runtime/preflight")
def runtime_preflight(task: str, device: str = "cuda:0"):
    """Read-only launch diagnostics using the same policy as managed runs."""
    known_tasks = {item["id"]: item for item in task_catalog()}
    task_info = known_tasks.get(task)
    if task not in task_ids() or task_info is None:
        return {
            "status": "blocked",
            "allowed": False,
            "task": {"id": task, "label": None},
            "requested_device": device,
            "runtime": {"execution_root": str(CONFIG.repo_root)},
            "blockers": [f"unsupported task: {task}"],
        }
    try:
        runtime = runtime_identity(CONFIG.repo_root, requested_device=device)
    except (RuntimePolicyError, ValueError, OSError) as error:
        return {
            "status": "blocked",
            "allowed": False,
            "task": {"id": task, "label": task_info["label"]},
            "requested_device": device,
            "runtime": {
                "kind": os.environ.get("ASCENTO_RUNTIME_KIND", "checkout"),
                "execution_root": str(CONFIG.repo_root),
                "source_revision": current_repository_version(),
            },
            "blockers": [str(error)],
        }
    return {
        "status": "ready",
        "allowed": True,
        "task": {"id": task, "label": task_info["label"]},
        "requested_device": device,
        "runtime": {**runtime, "source_revision": current_repository_version()},
        "blockers": [],
    }


@app.get("/api/runtime/identity")
def runtime_identity_report():
    report = runtime_revision_report()
    rows = _indexed_summaries()
    latest = rows[0] if rows else None
    repository = (latest or {}).get("repository_version") or {}
    report["latest_indexed_run"] = (
        {
            "id": latest.get("id"),
            "name": latest.get("display_name") or latest.get("name"),
            "state": latest.get("state"),
            "commit": repository.get("run_commit"),
            "status": repository.get("status", "unknown"),
        }
        if latest
        else None
    )
    return report


def _supervised_activity_runs(active_runs: list[dict]) -> list[dict]:
    root = ARTIFACT_ROOT.resolve()
    results: list[dict] = []
    for item in active_runs[:32]:
        relative = item.get("path")
        if not isinstance(relative, str):
            continue
        run_dir = (root / relative).resolve()
        try:
            run_dir.relative_to(root)
        except ValueError:
            continue
        status_path = run_dir / "run_status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
        status = status if isinstance(status, dict) else {}
        telemetry_path = run_dir / "telemetry.jsonl"
        try:
            heartbeat_at = telemetry_path.stat().st_mtime
        except OSError:
            heartbeat_at = None
        process = process_status(status.get("pid"), status.get("pid_namespace"))
        results.append(
            {
                **item,
                "started_at": status.get("started_at"),
                "heartbeat_at": heartbeat_at,
                "process": process,
            }
        )
    return results


@app.get("/api/activity")
def activity_snapshot():
    """Return bounded activity with per-source verification and freshness."""
    checked_at = time.time()
    indexed_rows = _annotated_summaries()
    active_states = {"starting", "running", "stopping"}
    indexed_active = [
        {
            "id": row.get("id"),
            "name": row.get("display_name") or row.get("name"),
            "state": row.get("state"),
            "updated_at": row.get("updated_at"),
            "stale": row.get("stale"),
            "source": "artifact-index",
        }
        for row in indexed_rows
        if row.get("state") in active_states
    ][:16]
    try:
        supervisor_status = SUPERVISOR.status()
        supervised_runs = _supervised_activity_runs(
            supervisor_status.get("active_runs")
            if isinstance(supervisor_status.get("active_runs"), list)
            else []
        )
        supervisor_available = True
        supervisor_error = None
    except SupervisorUnavailable as error:
        supervised_runs = []
        supervisor_available = False
        supervisor_error = str(error)

    if not supervisor_available:
        trainer = {
            "status": "unknown",
            "verified": False,
            "source": "host-supervisor",
            "active_runs": None,
            "indexed_active_runs": indexed_active,
            "message": supervisor_error,
        }
    else:
        process_verified = all(
            run.get("process", {}).get("alive") is not None for run in supervised_runs
        )
        trainer = {
            "status": "active" if supervised_runs else "idle",
            "verified": process_verified,
            "source": "host-supervisor+run-status",
            "active_runs": supervised_runs,
            "indexed_active_runs": indexed_active,
        }

    try:
        viewer_rows = VIEWER_SERVICE.list().get("viewers", [])
    except (OSError, RuntimeError) as error:
        viewer_rows = None
        viewer = {"status": "unknown", "verified": False, "message": str(error), "items": None}
    else:
        viewer_active = [item for item in viewer_rows if item.get("state") in active_states]
        viewer = {
            "status": "active" if viewer_active else "idle",
            "verified": True,
            "source": "managed-viewer-service",
            "items": viewer_rows[:8],
        }

    database = DATABASE.status()
    database_status = "optional" if not database.get("enabled") else (
        "healthy" if database.get("available") else "degraded"
    )
    gpu = gpu_snapshot()
    return {
        "checked_at": checked_at,
        "fresh_for_seconds": 15,
        "trainer": trainer,
        "evaluator": {
            "status": "unknown",
            "verified": False,
            "source": None,
            "message": "no evaluator process registry is configured",
        },
        "viewer": viewer,
        "render": {
            "status": "unknown",
            "verified": False,
            "source": None,
            "message": "render jobs are not tracked by the host supervisor",
        },
        "gpu": {"observed_at": checked_at, **gpu},
        "index": {
            "status": database_status,
            "backend": database.get("backend"),
            "available": database.get("available"),
            "last_successful_sync_at": database.get("last_successful_sync_at"),
            "source_conflicts": database.get("source_conflicts"),
            "indexed_run_count": len(indexed_rows),
            "source": "postgresql" if database.get("available") else "filesystem",
        },
    }


@app.get("/api/evaluation-suites")
def evaluation_suites():
    return {
        "suites": discover_suites(EVALUATION_SUITE_ROOT, EVALUATION_ROOT),
        "read_only": True,
        "evaluation_launch_available": False,
    }


@app.get("/api/assessments")
def assessments():
    """Return bounded, deterministic read-only operational and evidence findings."""
    checked_at = time.time()
    rows = _indexed_summaries()
    latest = rows[0] if rows else None
    run: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    if latest and latest.get("id"):
        run_id = str(latest["id"])
        try:
            run = RUN_SERVICE.progress(run_id)
            checkpoint_rows = VIEWER_SERVICE.checkpoints(run_id).get("checkpoints") or []
            stable_rows = [item for item in checkpoint_rows if item.get("stable") is True]
            if stable_rows:
                checkpoint = checkpoint_evidence(
                    run_id=run_id,
                    run_dir=RUN_SERVICE.resolve(run_id).path,
                    stable_checkpoints=stable_rows,
                    evaluation_root=EVALUATION_ROOT,
                )
        except (KeyError, OSError, ValueError, RuntimeError):
            checkpoint = None
    evaluation_rows = _evaluation_summaries()[:250]
    components = _system_components(checked_at=checked_at)
    return {
        "assessed_at": checked_at,
        "read_only": True,
        "assessments": build_assessments(
            checked_at=checked_at,
            components=components,
            database=DATABASE.status(),
            runtime=runtime_revision_report(),
            activity=activity_snapshot(),
            run=run,
            checkpoint=checkpoint,
            evaluations=evaluation_rows,
        ),
    }


@app.get("/api/experiments")
def experiments():
    return discover_experiments(EXPERIMENT_ROOT, _annotated_summaries())


@app.get("/api/experiments/{experiment_id:path}")
def experiment(experiment_id: str):
    payload = discover_experiments(EXPERIMENT_ROOT, _annotated_summaries())
    match = next((item for item in payload["programs"] if item["id"] == experiment_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="experiment plan not found")
    return match


@app.get("/api/evaluations")
def evaluations(
    request: Request,
    task: str | None = None,
    suite: str | None = None,
    status: str | None = None,
    evidence_class: str | None = None,
    experiment_id: str | None = None,
    checkpoint_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    rows = _evaluation_summaries()
    if task:
        rows = [row for row in rows if row.get("task") == task]
    if suite:
        rows = [row for row in rows if row.get("suite_id") == suite]
    if status:
        rows = [row for row in rows if row.get("status") == status.upper()]
    if evidence_class:
        rows = [row for row in rows if row.get("evidence_class") == evidence_class]
    if experiment_id:
        rows = [row for row in rows if row.get("experiment_id") == experiment_id]
    if checkpoint_id:
        rows = [
            row for row in rows
            if checkpoint_id in str(row.get("checkpoint") or "")
            or checkpoint_id in str(row.get("checkpoint_sha256") or "")
        ]
    total = len(rows)
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    payload = {
        "evaluations": rows[safe_offset : safe_offset + safe_limit],
        "total": total,
        "limit": safe_limit,
        "offset": safe_offset,
    }
    etag = _evaluation_etag(payload)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return JSONResponse(payload, headers={"ETag": etag, "Cache-Control": "private, max-age=0, must-revalidate"})


@app.get("/api/evaluations/{evaluation_id:path}/gates")
def evaluation_gates(evaluation_id: str):
    try:
        value = evaluation_detail(EVALUATION_ROOT, evaluation_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    gate = value.get("gate") or {}
    return {
        "evaluation_id": evaluation_id,
        "status": (value.get("evaluation") or {}).get("status", "INCOMPLETE"),
        "gates": gate.get("gates", []) if isinstance(gate, dict) else [],
    }


@app.get("/api/evaluations/compare")
def evaluation_comparison(baseline: str, candidate: str):
    try:
        return compare_evaluations(EVALUATION_ROOT, baseline, candidate)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/evaluations/{evaluation_id:path}/scenarios")
def evaluation_scenario_page(
    evaluation_id: str,
    metric: str | None = None,
    direction: str = "high",
    limit: int = 100,
    offset: int = 0,
):
    try:
        return evaluation_scenarios(
            EVALUATION_ROOT,
            evaluation_id,
            metric=metric,
            direction=direction,
            limit=limit,
            offset=offset,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/evaluations/{evaluation_id:path}")
def evaluation(evaluation_id: str):
    try:
        return evaluation_detail(EVALUATION_ROOT, evaluation_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/blender/renders")
def blender_renders(limit: int = 100):
    """List render manifests and safe links to media under the mounted captures tree."""
    limit = max(1, min(limit, 500))
    root = BLENDER_RENDER_ROOT.resolve()
    if not root.is_dir():
        return {"renders": [], "root": "captures/blender"}

    try:
        manifests = sorted(
            (path for path in root.rglob("*.manifest.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:limit]
    except OSError as error:
        raise HTTPException(
            status_code=500, detail=f"could not scan Blender renders: {error}"
        ) from error

    renders = []
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        try:
            manifest_relative = manifest_path.relative_to(root).as_posix()
        except ValueError:
            continue
        manifest["manifest_path"] = manifest_relative
        manifest["manifest_url"] = _blender_asset_url(manifest_path, root)
        outputs = manifest.get("outputs")
        if isinstance(outputs, dict):
            for output_name in ("blend", "video", "preview"):
                output = outputs.get(output_name)
                if not isinstance(output, dict) or not isinstance(output.get("path"), str):
                    continue
                output["url"] = _blender_asset_url(manifest_path.parent / output["path"], root)
        renders.append(manifest)
    return {"renders": renders, "root": "captures/blender"}


@app.get("/api/blender/renders/files/{asset_path:path}")
def blender_render_file(asset_path: str):
    root = BLENDER_RENDER_ROOT.resolve()
    candidate = (root / Path(asset_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="render asset not found") from error
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="render asset not found")
    if candidate.suffix.lower() not in BLENDER_RENDER_ASSET_SUFFIXES and not candidate.name.endswith(
        ".manifest.json"
    ):
        raise HTTPException(status_code=404, detail="render asset type is not served")
    return FileResponse(candidate)


@app.get("/api/config")
def configuration():
    return {
        **CONFIG.public_dict(),
        "startup_warnings": STARTUP_WARNINGS,
        "repository_version": current_repository_version(),
    }


@app.get("/api/system")
def system_status(refresh: bool = False):
    """Return host repository/update/Tailnet state through the restricted supervisor."""
    try:
        return {"connected": True, **SUPERVISOR.status(refresh=refresh)}
    except SupervisorUnavailable as error:
        return {
            "connected": False,
            "error": str(error),
            "repository": None,
            "active_runs": None,
            "update": {"status": "unavailable"},
            "tailscale": {
                "status": "unknown",
                "enabled": None,
                "connected": None,
                "error": "cannot verify without the host supervisor",
            },
            "can_update": False,
            "update_blockers": ["host supervisor is unavailable"],
        }


@app.post("/api/system/update", status_code=202)
def system_update(request: Request):
    """Ask the host supervisor to update to the newest origin/main."""
    _require_control_session(request)
    try:
        result = SUPERVISOR.update()
        DATABASE.record_event("system_update_requested", "Canonical checkout update requested through the dashboard")
        return result
    except SupervisorUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except SupervisorRejected as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/tasks")
def tasks():
    return {"tasks": task_catalog()}


@app.get("/api/runs/index")
def run_index():
    """Small run-list payload for the control-room UI."""
    try:
        rows = _indexed_summaries()
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"failed to scan artifact root {ARTIFACT_ROOT}: {error}",
        ) from error
    return {"runs": rows}


@app.get("/api/overview")
def overview():
    """Return everything the landing page needs in one bounded request."""
    rows = _indexed_summaries()
    active_states = {"starting", "running", "stopping"}
    active = next((row for row in rows if row.get("state") in active_states), None)
    counts = {
        "total": len(rows),
        "active": sum(1 for row in rows if row.get("state") in active_states),
        "errors": sum(1 for row in rows if row.get("state") == "error"),
        "outdated": sum(
            1 for row in rows if (row.get("repository_version") or {}).get("is_outdated")
        ),
    }
    if active is None:
        return {
            "active_run": None,
            "recent_run": rows[0] if rows else None,
            "curriculum": None,
            "series": [],
            "events": DATABASE.recent_events(limit=12),
            "counts": counts,
            "database": DATABASE.status(),
        }

    try:
        detail, curriculum = _curriculum_snapshot(str(active["id"]))
        current = _overview_run(detail)
        DATABASE.sync_run(current, curriculum)
        return {
            "active_run": current,
            "recent_run": rows[0] if rows else None,
            "curriculum": curriculum,
            "series": _overview_series(str(active["id"])),
            "events": DATABASE.recent_events(run_id=str(active["id"]), limit=12),
            "counts": counts,
            "database": DATABASE.status(),
        }
    except (KeyError, OSError) as error:
        raise HTTPException(status_code=500, detail=f"failed to build overview: {error}") from error


@app.get("/api/runs")
def runs():
    try:
        summaries = _annotated_summaries()
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"failed to scan artifact root {ARTIFACT_ROOT}: {error}",
        ) from error
    return {"runs": summaries}


@app.post("/api/runs", status_code=202)
def create_run(request: RunCreateRequest, http_request: Request):
    _require_control_session(http_request)
    try:
        if request.experiment_id:
            known = discover_experiments(EXPERIMENT_ROOT, _annotated_summaries())
            if not any(item.get("id") == request.experiment_id for item in known.get("programs", [])):
                raise ValueError(f"unknown experiment plan: {request.experiment_id}")
        payload = request.model_dump()
        if request.parent_checkpoint:
            if not request.parent_run_id:
                raise ValueError("a parent run is required when selecting a parent checkpoint")
            checkpoint_path, _selected, _parent_dir = _resolve_stable_checkpoint(
                request.parent_run_id, request.parent_checkpoint
            )
            validate_checkpoint_for_task(checkpoint_path, request.task)
            payload["parent_checkpoint"] = str(checkpoint_path)
        created = RUN_SERVICE.create(payload)
        _invalidate_summary_cache()
        DATABASE.record_event(
            "run_started",
            f"{created.get('display_name') or created.get('name')}: training started",
            run_id=str(created.get("id")) if created.get("id") else None,
            payload={"task": request.task},
        )
        return created
    except KeyError as error:
        raise HTTPException(
            status_code=400, detail=f"parent run not found: {error.args[0]}"
        ) from error
    except (ValueError, PermissionError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/compare")
def compare_runs(run_ids: str):
    ids = [value.strip() for value in run_ids.split(",") if value.strip()]
    try:
        return RUN_SERVICE.compare(ids)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail=f"training run not found: {error.args[0]}"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    try:
        return RUN_SERVICE.detail(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error


@app.get("/api/runs/{run_id}/curriculum")
def run_curriculum(run_id: str):
    try:
        _, curriculum = _curriculum_snapshot(run_id)
        return {"curriculum": curriculum}
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error


@app.get("/api/runs/{run_id}/progress")
def run_progress(run_id: str):
    """Return the latest live snapshot without scanning detailed run history."""
    try:
        return RUN_SERVICE.progress(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error


@app.patch("/api/runs/{run_id}")
def update_run(run_id: str, request: RunUpdateRequest, http_request: Request):
    _require_control_session(http_request)
    try:
        updated = RUN_SERVICE.update_metadata(run_id, request.model_dump(exclude_unset=True))
        _invalidate_summary_cache()
        DATABASE.record_event(
            "metadata_updated",
            f"{updated.get('display_name') or updated.get('name')}: metadata updated",
            run_id=run_id,
        )
        return updated
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="training run or parent run not found"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/runs/{run_id}/stop", status_code=202)
def stop_run(run_id: str, request: RunStopRequest, http_request: Request):
    _require_control_session(http_request)
    try:
        stopped = RUN_SERVICE.stop(run_id, reason=request.reason.strip() or "user_requested")
        _invalidate_summary_cache()
        DATABASE.record_event(
            "stop_requested",
            f"{stopped.get('display_name') or stopped.get('name')}: graceful stop requested",
            run_id=run_id,
            payload={"reason": request.reason.strip() or "user_requested"},
        )
        return stopped
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except ProcessLookupError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/runs/{run_id}/checkpoints")
def run_checkpoints(run_id: str):
    try:
        result = VIEWER_SERVICE.checkpoints(run_id)
        DATABASE.sync_checkpoints(run_id, result.get("checkpoints") or [])
        return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


def validate_checkpoint_for_task(checkpoint: Path, task: str) -> None:
    """Use the same strict contract validator as the managed-run launcher."""
    from dashboard.launch import _validate_parent_checkpoint

    _validate_parent_checkpoint(checkpoint, task)


def _resolve_stable_checkpoint(run_id: str, relative_path: str) -> tuple[Path, dict, Path]:
    ref = _run(run_id)
    stable_rows = [
        row for row in VIEWER_SERVICE.checkpoints(run_id).get("checkpoints", [])
        if isinstance(row, dict) and row.get("stable") is True
    ]
    selected = next((row for row in stable_rows if row.get("relative_path") == relative_path), None)
    if selected is None:
        raise ValueError("selected checkpoint is not a published stable checkpoint for this run")
    run_dir = Path(ref.path).resolve()
    checkpoint_path = (run_dir / str(selected["relative_path"])).resolve()
    try:
        checkpoint_path.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("checkpoint escapes the managed run") from error
    if not checkpoint_path.is_file():
        raise ValueError("stable checkpoint file is missing")
    return checkpoint_path, selected, run_dir


@app.get("/api/runs/{run_id}/checkpoint-compatibility")
def run_checkpoint_compatibility(run_id: str, checkpoint: str, task: str):
    """Preflight one published checkpoint against a declared task contract."""
    if task not in task_ids():
        raise HTTPException(status_code=422, detail=f"unsupported task: {task}")
    try:
        checkpoint_path, selected, run_dir = _resolve_stable_checkpoint(run_id, checkpoint)
    except (OSError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=f"cannot inspect stable checkpoints: {error}") from error
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        evidence = checkpoint_evidence(
            run_id=run_id,
            run_dir=run_dir,
            stable_checkpoints=[selected],
            evaluation_root=EVALUATION_ROOT,
        )
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=404, detail=f"cannot verify checkpoint artifact: {error}") from error

    try:
        validate_checkpoint_for_task(checkpoint_path, task)
    except ValueError as error:
        return {
            "run_id": run_id,
            "task": task,
            "checkpoint": selected["relative_path"],
            "checkpoint_sha256": evidence["sha256"],
            "status": "INCOMPATIBLE",
            "compatible": False,
            "reason": str(error),
        }
    except (OSError, RuntimeError, ImportError) as error:
        return {
            "run_id": run_id,
            "task": task,
            "checkpoint": selected["relative_path"],
            "checkpoint_sha256": evidence["sha256"],
            "status": "UNVERIFIABLE",
            "compatible": False,
            "reason": f"checkpoint compatibility could not be verified: {error}",
        }
    return {
        "run_id": run_id,
        "task": task,
        "checkpoint": selected["relative_path"],
        "checkpoint_sha256": evidence["sha256"],
        "status": "COMPATIBLE",
        "compatible": True,
        "reason": "plant, action, and task contracts match the selected task",
    }


@app.get("/api/runs/{run_id}/checkpoint-evidence")
def run_checkpoint_evidence(run_id: str, checkpoint: str = "latest"):
    try:
        ref = RUN_SERVICE.resolve(run_id)
        available = VIEWER_SERVICE.checkpoints(run_id).get("checkpoints") or []
        return checkpoint_evidence(
            run_id=run_id,
            run_dir=ref.path,
            stable_checkpoints=available,
            evaluation_root=EVALUATION_ROOT,
            checkpoint=checkpoint,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/{run_id}/architecture")
def run_architecture(run_id: str):
    try:
        return VIEWER_SERVICE.architecture(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except OSError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/viewers")
def viewers():
    return VIEWER_SERVICE.list()


@app.post("/api/viewers", status_code=202)
def start_viewer(payload: ViewerCreateRequest, request: Request):
    _require_control_session(request)
    try:
        result = VIEWER_SERVICE.start(
            run_id=payload.run_id,
            checkpoint=payload.checkpoint,
            follow=payload.follow,
            device=payload.device,
            jacobian_hz=payload.jacobian_hz,
        )
        DATABASE.record_event(
            "viewer_started",
            f"Viewer started for run {payload.run_id}",
            run_id=payload.run_id,
            payload={"viewer_id": result.get("id"), "checkpoint": payload.checkpoint, "follow": payload.follow},
        )
        return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ViewerBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=503, detail=f"viewer could not start: {error}") from error


@app.get("/api/viewers/{viewer_id}")
def viewer_status(viewer_id: str):
    try:
        return VIEWER_SERVICE.get(viewer_id)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.get("/api/viewers/{viewer_id}/logs")
def viewer_logs(viewer_id: str, tail: int = 300):
    try:
        return VIEWER_SERVICE.logs(viewer_id, tail=tail)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.get("/api/viewers/{viewer_id}/waypoints")
def viewer_waypoints(viewer_id: str):
    try:
        return VIEWER_SERVICE.waypoint_state(viewer_id)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.post("/api/viewers/{viewer_id}/waypoints", status_code=202)
def command_viewer_waypoint(
    viewer_id: str, payload: ViewerWaypointRequest, request: Request
):
    _require_control_session(request)
    try:
        result = VIEWER_SERVICE.command_waypoint(viewer_id, payload.model_dump())
        DATABASE.record_event(
            "viewer_waypoint_command",
            f"Waypoint command {payload.operation} accepted by viewer {viewer_id}",
            run_id=str(result.get("run_id")) if result.get("run_id") else None,
            payload={"viewer_id": viewer_id, "request_id": result.get("request_id"), "command": payload.model_dump()},
        )
        return result
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error
    except ViewerBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/viewers/{viewer_id}/introspection/schema")
def viewer_introspection_schema(viewer_id: str):
    try:
        return VIEWER_SERVICE.introspection_schema(viewer_id)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.get("/api/viewers/{viewer_id}/introspection/latest")
def viewer_introspection_latest(viewer_id: str):
    try:
        return VIEWER_SERVICE.introspection_latest(viewer_id)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.get("/api/viewers/{viewer_id}/introspection/captures")
@app.get("/api/viewers/{viewer_id}/captures")
def viewer_introspection_captures(viewer_id: str, limit: int = 50):
    try:
        return VIEWER_SERVICE.introspection_captures(viewer_id, limit=limit)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.get("/api/viewers/{viewer_id}/introspection/captures/{event_id}")
@app.get("/api/viewers/{viewer_id}/captures/{event_id}")
def viewer_introspection_capture(viewer_id: str, event_id: str):
    try:
        return VIEWER_SERVICE.introspection_capture(viewer_id, event_id)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="capture not found") from error


@app.post("/api/viewers/{viewer_id}/introspection/captures", status_code=202)
@app.post("/api/viewers/{viewer_id}/captures", status_code=202)
def request_viewer_introspection_capture(viewer_id: str, request: Request):
    _require_control_session(request)
    try:
        result = VIEWER_SERVICE.request_introspection_capture(viewer_id)
        DATABASE.record_event(
            "viewer_capture_requested",
            f"Policy introspection capture requested for viewer {viewer_id}",
            payload={"viewer_id": viewer_id, "request_id": result.get("request_id")},
        )
        return result
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error
    except ViewerBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except CaptureBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/viewers/{viewer_id}/introspection/explanations", status_code=202)
@app.post("/api/viewers/{viewer_id}/explanations", status_code=202)
def request_viewer_introspection_explanation(
    viewer_id: str,
    payload: ViewerExplanationRequest,
    request: Request,
):
    _require_control_session(request)
    try:
        result = VIEWER_SERVICE.request_introspection_explanation(viewer_id, payload.model_dump())
        DATABASE.record_event(
            "viewer_explanation_requested",
            f"Policy introspection explanation requested for viewer {viewer_id}",
            payload={
                "viewer_id": viewer_id,
                "explanation_id": result.get("explanation_id"),
                "checkpoint": payload.checkpoint,
                "action_index": payload.action_index,
            },
        )
        return result
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error
    except ViewerBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ExplanationBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/viewers/{viewer_id}/introspection/explanations/{explanation_id}")
@app.get("/api/viewers/{viewer_id}/explanations/{explanation_id}")
def viewer_introspection_explanation(viewer_id: str, explanation_id: str):
    try:
        return VIEWER_SERVICE.introspection_explanation(viewer_id, explanation_id)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.get("/api/viewers/{viewer_id}/introspection/stream")
async def viewer_introspection_stream(viewer_id: str):
    try:
        VIEWER_SERVICE.get(viewer_id)
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error

    async def events():
        last_sequence = None
        idle_polls = 0
        while True:
            try:
                frame = VIEWER_SERVICE.introspection_latest(viewer_id)
            except ViewerNotFoundError:
                return
            sequence = frame.get("sequence_id")
            if isinstance(sequence, int) and sequence != last_sequence:
                payload = json.dumps(frame, separators=(",", ":"), allow_nan=False)
                yield f"id: {sequence}\nevent: frame\ndata: {payload}\n\n"
                last_sequence = sequence
                idle_polls = 0
            else:
                idle_polls += 1
                if idle_polls % 150 == 0:
                    yield ": keep-alive\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/api/viewers/{viewer_id}", status_code=202)
def stop_viewer(viewer_id: str, request: Request):
    _require_control_session(request)
    try:
        result = VIEWER_SERVICE.stop(viewer_id)
        DATABASE.record_event(
            "viewer_stop_requested",
            f"Viewer stop requested for {viewer_id}",
            run_id=str(result.get("run_id")) if result.get("run_id") else None,
            payload={"viewer_id": viewer_id},
        )
        return result
    except ViewerNotFoundError as error:
        raise HTTPException(status_code=404, detail="viewer not found") from error


@app.get("/api/runs/{run_id}/summary.json")
def run_summary(run_id: str):
    try:
        summary = RUN_SERVICE.detail(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="training run not found") from error
    return JSONResponse(
        content=summary,
        headers={"Content-Disposition": 'attachment; filename="run-summary.json"'},
    )


@app.get("/api/runs/{run_id}/telemetry")
def telemetry(run_id: str, limit: int = 2000, max_points: int | None = None):
    ref = _run(run_id)
    if max_points is not None:
        max_points = max(2, min(max_points, 5000))
        raw_records = load_training_records(ref.path, limit=None)
        source_records = len(raw_records)
        records = decorate_records(_sample_records(raw_records, max_points), ref.path)
        return {
            "records": records,
            "source_records": source_records,
            "sampled": source_records > len(records),
            "coverage": _telemetry_coverage(records),
        }
    limit = max(1, min(limit, 20_000))
    return {"records": load_dashboard_records(ref.path, limit=limit)}


@app.get("/api/runs/{run_id}/logs")
def logs(run_id: str, tail: int = 500):
    ref = _run(run_id)
    tail = max(1, min(tail, 5000))
    log_path = training_log_path(ref.path, ARTIFACT_ROOT)
    return {"lines": [line.rstrip("\n") for line in tail_lines(log_path, tail)]}


@app.get("/api/runs/{run_id}/logs/stream")
async def stream_logs(run_id: str, request: Request):
    ref = _run(run_id)
    log_path = training_log_path(ref.path, ARTIFACT_ROOT)

    async def event_stream():
        try:
            position = log_path.stat().st_size
        except OSError:
            position = 0
        inode = None
        while True:
            if await request.is_disconnected():
                return
            try:
                stat = log_path.stat()
                current_inode = getattr(stat, "st_ino", None)
                if inode is not None and current_inode != inode:
                    position = 0
                inode = current_inode
                if stat.st_size < position:
                    position = 0
                with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                for line in chunk.splitlines():
                    yield f"data: {json.dumps({'line': line})}\n\n"
            except FileNotFoundError:
                pass
            except OSError as error:
                yield (f"event: monitor_error\ndata: {json.dumps({'message': str(error)})}\n\n")
            yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return JSONResponse(
        {
            "message": "Dashboard frontend is not built yet.",
            "build": "cd dashboard/frontend && npm install && npm run build",
            "api": "/api/runs",
            "health": "/api/health",
            "config": "/api/config",
            "system": "/api/system",
        }
    )


if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend_route(path: str):
    """Serve the SPA shell for real browser routes.

    API misses stay API 404s instead of returning HTML. The static assets mount
    is registered before this fallback and therefore keeps handling /assets.
    """
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    index_path = FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Dashboard frontend is not built yet")


@app.on_event("startup")
def initialize_dashboard_database() -> None:
    DATABASE.initialize()
    if DATABASE.enabled and DATABASE.error:
        warning = f"Dashboard database unavailable; using filesystem fallback: {DATABASE.error}"
        if warning not in STARTUP_WARNINGS:
            STARTUP_WARNINGS.append(warning)


@app.on_event("shutdown")
def stop_managed_viewers() -> None:
    VIEWER_SERVICE.stop_all()
    DATABASE.dispose()

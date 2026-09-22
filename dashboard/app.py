"""FastAPI service for remotely monitoring and managing Ascento PPO training."""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard.config import load_config, validate_startup
from dashboard.curriculum import curriculum_for_run
from dashboard.database import DashboardDatabase
from dashboard.health import (
    build_run_info,
    decorate_records,
    discover_dashboard_runs,
    list_dashboard_summaries,
    load_dashboard_records,
)
from dashboard.monitor import load_training_records, tail_lines, training_log_path
from dashboard.run_service import RunService
from dashboard.supervisor_client import (
    SupervisorClient,
    SupervisorRejected,
    SupervisorUnavailable,
)
from dashboard.task_catalog import task_catalog
from dashboard.versioning import current_repository_version
from dashboard.viewer_service import ViewerBusyError, ViewerNotFoundError, ViewerService

CONFIG = load_config()
STARTUP_WARNINGS = validate_startup(CONFIG, create_artifact_root=False)
ARTIFACT_ROOT = CONFIG.artifact_root
FRONTEND_DIST = CONFIG.frontend_dist
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

app = FastAPI(title="Ascento Control", version="3.0")


class RunCreateRequest(BaseModel):
    display_name: str
    task: str = "Ascento-Balance-Flat"
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    purpose: str = ""
    parent_run_id: str | None = None
    parent_checkpoint: str | None = None
    episode_horizon_s: float | None = Field(default=None)
    allow_dirty_provenance: bool = False
    training_args: list[str] = Field(default_factory=list)


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
    return [
        records[round(index * last_index / (max_points - 1))]
        for index in range(max_points)
    ]


def _telemetry_coverage(records: list[dict]) -> dict[str, dict[str, int]]:
    """Describe which canonical series are genuinely available after sampling."""
    keys = sorted(
        {
            str(key)
            for record in records
            for key in (record.get("canonical_metrics") or {})
        }
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


def _invalidate_summary_cache() -> None:
    global _SUMMARY_CACHE, _INDEX_CACHE
    with _SUMMARY_CACHE_LOCK:
        _SUMMARY_CACHE = None
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE = None


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
    problems = _artifact_health()
    # Health is polled alongside /api/runs.  Do not make it perform a second
    # full artifact traversal just to produce an informational count.
    run_count = _cached_summary_count()
    if run_count is None and not ARTIFACT_ROOT.exists():
        run_count = 0
    return {
        "ok": not problems,
        "status": "ok" if not problems else "error",
        "checked_at": time.time(),
        "artifact_root": str(ARTIFACT_ROOT),
        "run_count": run_count,
        "problems": problems,
        "warnings": STARTUP_WARNINGS,
        "config": CONFIG.public_dict(),
        "repository_version": current_repository_version(),
        "database": DATABASE.status(),
    }


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
            "active_runs": [],
            "update": {"status": "unavailable"},
            "tailscale": {"enabled": False, "connected": False},
            "can_update": False,
            "update_blockers": ["host supervisor is unavailable"],
        }


@app.post("/api/system/update", status_code=202)
def system_update(request: Request):
    """Ask the host supervisor to update to the newest origin/main."""
    # Requiring a non-simple custom header prevents cross-site forms and simple
    # browser requests from invoking the privileged host boundary. Tailnet
    # grants/ACLs remain the authentication boundary for reaching this service.
    if request.headers.get("x-ascento-control") != "1":
        raise HTTPException(status_code=403, detail="missing dashboard control header")
    try:
        return SUPERVISOR.update()
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
            1
            for row in rows
            if (row.get("repository_version") or {}).get("is_outdated")
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
def create_run(request: RunCreateRequest):
    try:
        created = RUN_SERVICE.create(request.model_dump())
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
def update_run(run_id: str, request: RunUpdateRequest):
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
def stop_run(run_id: str, request: RunStopRequest):
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
    if request.headers.get("x-ascento-control") != "1":
        raise HTTPException(status_code=403, detail="missing dashboard control header")
    try:
        return VIEWER_SERVICE.start(
            run_id=payload.run_id,
            checkpoint=payload.checkpoint,
            follow=payload.follow,
            device=payload.device,
        )
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


@app.delete("/api/viewers/{viewer_id}", status_code=202)
def stop_viewer(viewer_id: str, request: Request):
    if request.headers.get("x-ascento-control") != "1":
        raise HTTPException(status_code=403, detail="missing dashboard control header")
    try:
        return VIEWER_SERVICE.stop(viewer_id)
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

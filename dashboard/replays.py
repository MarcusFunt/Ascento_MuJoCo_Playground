"""Read-only replay discovery and opt-in Rerun export for the Analyze page."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse

from dashboard.config import load_config

router = APIRouter(prefix="/api/replays", tags=["replays"])
CAPTURE_ROOT = load_config().repo_root / "captures"
_EXPORT_LOCK = threading.Lock()
MAX_EXPORT_BYTES = 512 * 1024 * 1024


def _safe_file(relative: str, suffix: str) -> Path:
    """Constrain paths to the capture tree, blocking traversal and symlinks."""
    root = CAPTURE_ROOT.resolve()
    if not relative or "\x00" in relative or Path(relative).is_absolute():
        raise HTTPException(404, "Replay not found")
    candidate = root / relative
    try:
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ValueError("Outside capture root")
        # Disallow symlinks anywhere in the relative path.
        if any(part.is_symlink() for part in [candidate, *candidate.parents] if part != root):
            raise ValueError("Symlinked capture path")
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "Replay not found") from exc
    if candidate.suffix.lower() != suffix or not candidate.is_file():
        raise HTTPException(404, "Replay not found")
    return candidate


def _url(relative: str) -> str:
    return "/api/replays/files/" + quote(relative, safe="/")


@router.get("")
def list_replays(limit: int = 100):
    """Discover locally captured NPZ and generated RRD recordings."""
    limit = max(1, min(limit, 200))
    root = CAPTURE_ROOT.resolve()
    if not root.is_dir():
        return {"replays": []}
    recordings: dict[str, dict] = {}
    for suffix in (".npz", ".rrd"):
        for path in root.rglob(f"*{suffix}"):
            try:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or not path.resolve().is_relative_to(root)
                ):
                    continue
                relative = path.relative_to(root).as_posix()
                stem = relative[: -len(suffix)]
                entry = recordings.setdefault(
                    stem, {"name": stem, "source": None, "rrd_url": None, "updated_at": 0.0}
                )
                entry["updated_at"] = max(entry["updated_at"], path.stat().st_mtime)
                if suffix == ".npz":
                    entry["source"] = relative
                else:
                    entry["rrd_url"] = _url(relative)
            except (ValueError, OSError):
                continue
    return {
        "replays": sorted(recordings.values(), key=lambda entry: entry["updated_at"], reverse=True)[
            :limit
        ]
    }


@router.get("/files/{recording:path}")
def replay_file(recording: str):
    """Stream an RRD from the authorized capture directory; supports HTTP ranges."""
    path = _safe_file(recording, ".rrd")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, max-age=60", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/export/{recording:path}")
def export_replay(recording: str, x_ascento_control: str | None = Header(default=None)):
    """Convert an existing NPZ into Rerun format without invoking any training code."""
    if x_ascento_control != "1":
        raise HTTPException(403, "Explicit control header required")
    source = _safe_file(recording, ".npz")
    if source.stat().st_size > MAX_EXPORT_BYTES:
        raise HTTPException(413, "Capture too large to convert through the dashboard")
    target = source.with_suffix(".rrd")
    if not _EXPORT_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another replay export is in progress")
    try:
        if not target.is_file():
            try:
                from ascento_mjlab.tools.export_rerun import export_rerun

                export_rerun(source, target)
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
            except (ValueError, OSError) as exc:
                raise HTTPException(422, f"Cannot export this capture: {exc}") from exc
        relative = target.relative_to(CAPTURE_ROOT.resolve()).as_posix()
        return {"rrd_url": _url(relative), "name": relative, "bytes": target.stat().st_size}
    finally:
        _EXPORT_LOCK.release()

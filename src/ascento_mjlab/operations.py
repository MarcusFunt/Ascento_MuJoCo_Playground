"""Shared, filesystem-safe operations for the Ascento CLI and MCP server.

This module deliberately keeps the high-level operational surface independent
of the dashboard HTTP API.  The web UI, ``ascento`` CLI, and MCP server can
therefore inspect the same immutable evaluation artifacts and resolve a managed
run's latest checkpoint in the same way.
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .plant_contract import current_plant_contract, plant_contracts_compatible


def repo_root() -> Path:
    """Return the checkout root without relying on the caller's CWD."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def ensure_checkout_import_path() -> Path:
    """Make checkout-local companion packages importable from console scripts.

    ``ascento`` and ``ascento-mcp`` are installed console entry points, so
    Python initializes ``sys.path[0]`` to ``.venv/bin`` rather than to the
    caller's current directory.  The dashboard is deliberately a companion
    package at the checkout root (it is also copied into the dashboard
    container), not part of the ``src/ascento_mjlab`` distribution.  Add the
    discovered checkout explicitly so the CLI and MCP server behave the same
    whether they are invoked from the repository root or elsewhere.
    """
    root = repo_root().resolve()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def artifact_root(value: str | Path | None = None) -> Path:
    """Resolve the shared managed-training artifact root."""
    raw = value or os.environ.get("ASCENTO_ARTIFACT_ROOT") or "logs/rsl_rl"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def evaluation_root(value: str | Path | None = None) -> Path:
    """Resolve the shared immutable-evaluation artifact root."""
    raw = value or os.environ.get("ASCENTO_EVALUATION_ROOT") or "evaluations"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


_CHECKPOINT_NUMBER = re.compile(r"(?:model|checkpoint)[_-]?(\d+)", re.IGNORECASE)


def latest_checkpoint(run_dir: Path) -> Path:
    """Find the newest numbered policy checkpoint in an already resolved run."""
    candidates = [
        path
        for path in run_dir.rglob("*.pt")
        if path.is_file() and (path.name.startswith("model_") or path.name.startswith("checkpoint_"))
    ]
    if not candidates:
        raise FileNotFoundError(f"no model_*.pt or checkpoint_*.pt files below {run_dir}")

    def key(path: Path) -> tuple[int, float]:
        match = _CHECKPOINT_NUMBER.search(path.stem)
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        return (int(match.group(1)) if match else -1, modified)

    return max(candidates, key=key).resolve()


def resolve_checkpoint(
    *,
    checkpoint: str | Path | None = None,
    run_id: str | None = None,
    artifacts: str | Path | None = None,
) -> Path:
    """Resolve one explicit checkpoint or the current checkpoint for a managed run."""
    if bool(checkpoint) == bool(run_id):
        raise ValueError("provide exactly one of checkpoint or run_id")
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = repo_root() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return path

    ensure_checkout_import_path()
    from dashboard.run_service import RunService

    root = artifact_root(artifacts)
    service = RunService(root)
    ref = service.resolve(str(run_id))
    detail = service.detail(str(run_id))
    run_info = detail.get("run_info") if isinstance(detail.get("run_info"), dict) else {}
    raw = run_info.get("checkpoint_path")
    candidates: list[Path] = []
    if isinstance(raw, str) and raw.strip():
        checkpoint_path = Path(raw)
        if checkpoint_path.is_absolute():
            candidates.append(checkpoint_path)
        else:
            candidates.extend((ref.path / checkpoint_path, ref.path.parent / checkpoint_path, root / checkpoint_path))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if _inside(resolved, root) and resolved.is_file():
            return resolved
    return latest_checkpoint(ref.path)


def default_capture_dir(checkpoint: Path) -> Path:
    """Create a collision-resistant default location for ad-hoc policy captures."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", checkpoint.stem).strip("-._") or "policy"
    return repo_root() / "captures" / f"{stamp}_{stem}"


def resolve_evaluation_dir(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a completed or partial evaluation directory below the output root."""
    base = evaluation_root(root)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    if not _inside(candidate, base):
        raise ValueError(f"evaluation directory must be below {base}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"evaluation directory does not exist: {candidate}")
    if not (candidate / "manifest.json").is_file() and not (candidate / "suite.json").is_file():
        raise ValueError(f"evaluation directory has neither manifest.json nor suite.json: {candidate}")
    return candidate


def _partial_report_row(directory: Path, base: Path) -> dict[str, Any]:
    suite = _json_object(directory / "suite.json") if (directory / "suite.json").is_file() else {}
    return {
        "id": directory.relative_to(base).as_posix(),
        "path": str(directory),
        "suite_id": suite.get("suite_id"),
        "task": suite.get("task"),
        "checkpoint": None,
        "checkpoint_sha256": None,
        "repository_commit": None,
        "repository_dirty": None,
        "started_at_utc": None,
        "finished_at_utc": None,
        "status": "INCOMPLETE",
        "scenario_count": None,
        "report_path": None,
        "clips_status": None,
        "plant_contract": None,
        "plant_compatibility": "legacy",
        "incomplete_reason": "manifest.json is missing; evaluation was interrupted or is still running",
    }


def _report_row(directory: Path, base: Path) -> dict[str, Any]:
    if not (directory / "manifest.json").is_file():
        return _partial_report_row(directory, base)
    manifest = _json_object(directory / "manifest.json")
    gate = _json_object(directory / "gate.json") if (directory / "gate.json").is_file() else {}
    clips = _json_object(directory / "clips_manifest.json") if (directory / "clips_manifest.json").is_file() else {}
    report = directory / "report.html"
    plant_contract = manifest.get("plant_contract")
    plant_compatibility = (
        "current"
        if plant_contracts_compatible(plant_contract, current_plant_contract())
        else "legacy"
        if plant_contract is None
        else "incompatible"
    )
    return {
        "id": directory.relative_to(base).as_posix(),
        "path": str(directory),
        "suite_id": manifest.get("suite_id"),
        "task": manifest.get("task"),
        "checkpoint": manifest.get("checkpoint"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "repository_commit": manifest.get("repository_commit"),
        "repository_dirty": manifest.get("repository_dirty"),
        "started_at_utc": manifest.get("started_at_utc"),
        "finished_at_utc": manifest.get("finished_at_utc"),
        "status": gate.get("status"),
        "scenario_count": manifest.get("scenario_count"),
        "report_path": str(report) if report.is_file() else None,
        "clips_status": clips.get("status") if clips else None,
        "plant_contract": plant_contract,
        "plant_compatibility": plant_compatibility,
    }


def list_evaluations(root: str | Path | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
    """List completed and incomplete evaluation artifacts without loading SQLite results."""
    base = evaluation_root(root)
    if not base.is_dir():
        return []
    reports: list[tuple[float, dict[str, Any]]] = []
    directories = {path.parent for path in base.rglob("manifest.json")}
    directories.update(path.parent for path in base.rglob("suite.json"))
    for directory in directories:
        if not _inside(directory, base):
            continue
        try:
            row = _report_row(directory, base)
            modified = max(
                (path.stat().st_mtime for path in (directory / "manifest.json", directory / "suite.json") if path.is_file()),
                default=0.0,
            )
        except (OSError, ValueError):
            continue
        reports.append((modified, row))
    reports.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in reports[: max(1, min(int(limit), 500))]]


def evaluation_details(value: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Read the compact, human-reviewable payload of one evaluation artifact."""
    base = evaluation_root(root)
    directory = resolve_evaluation_dir(value, base)
    payload = _report_row(directory, base)
    for name in ("summary", "gate", "consistency", "failures", "clips_manifest"):
        path = directory / f"{name}.json"
        payload[name] = _json_object(path) if path.is_file() else None
    suite_path = directory / "suite.json"
    payload["suite"] = _json_object(suite_path) if suite_path.is_file() else None
    payload["archive_path"] = str(directory.with_suffix(".zip")) if directory.with_suffix(".zip").is_file() else None
    return payload


def archive_evaluation(
    value: str | Path,
    *,
    root: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    """Create a recoverable ZIP copy of one immutable evaluation directory."""
    base = evaluation_root(root)
    directory = resolve_evaluation_dir(value, base)
    archive = Path(output).expanduser() if output else directory.with_suffix(".zip")
    if not archive.is_absolute():
        archive = base / archive
    archive = archive.resolve()
    if archive.suffix.lower() != ".zip":
        raise ValueError("archive output must have a .zip suffix")
    if not _inside(archive, base):
        raise ValueError(f"archive output must be below {base}")
    if _inside(archive, directory):
        raise ValueError("archive output cannot be inside the evaluation directory")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                handle.write(path, arcname=path.relative_to(directory).as_posix())
        temporary.replace(archive)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return archive


def list_evaluation_suites() -> list[dict[str, Any]]:
    """Return the immutable suite IDs that can be selected by evaluation commands."""
    from ascento_mjlab.evaluation.schema import load_suite

    suites: list[dict[str, Any]] = []
    root = repo_root() / "benchmarks" / "suites"
    for path in sorted(root.glob("*.toml")):
        try:
            suite = load_suite(path)
        except (OSError, ValueError):
            continue
        suites.append(
            {
                "id": suite.suite_id,
                "path": str(path),
                "task": suite.task,
                "policy_mode": suite.policy_mode,
                "family_count": len(suite.families),
                "scenario_count": sum(family.count for family in suite.families),
                "gate_count": len(suite.gates),
            }
        )
    return suites

"""Repository provenance helpers for dashboard run compatibility warnings."""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from dashboard.config import REPO_ROOT


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _parent_file(run_dir: Path, root: Path, filename: str) -> Path | None:
    root = root.resolve()
    for directory in (run_dir.resolve(), *run_dir.resolve().parents):
        try:
            directory.relative_to(root)
        except ValueError:
            break
        candidate = directory / filename
        if candidate.is_file():
            return candidate
        if directory == root:
            break
    return None


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


@lru_cache(maxsize=1)
def current_repository_version() -> dict[str, Any]:
    """Return the repository version represented by the running dashboard."""
    commit = os.environ.get("ASCENTO_REPOSITORY_COMMIT") or None
    branch = os.environ.get("ASCENTO_REPOSITORY_BRANCH") or None
    source = "environment" if commit else None

    state = _load_json(REPO_ROOT / ".maintenance" / "repository-version.json") or {}
    if not commit and state.get("commit"):
        commit = str(state["commit"])
        source = "maintenance-state"
    if not branch and state.get("branch"):
        branch = str(state["branch"])

    if not commit:
        commit = _git("rev-parse", "HEAD")
        if commit:
            source = "git"
    if not branch:
        branch = _git("branch", "--show-current")

    return {"commit": commit, "branch": branch, "source": source or "unknown"}


def run_repository_provenance(run_dir: Path, root: Path) -> dict[str, Any]:
    """Read exact or maintenance-inferred repository provenance for one run."""
    status_path = _parent_file(run_dir, root, "run_status.json")
    manifest_path = _parent_file(run_dir, root, "training_manifest.json")
    sidecar_path = _parent_file(run_dir, root, "repository_provenance.json")

    status = _load_json(status_path) if status_path else None
    manifest = _load_json(manifest_path) if manifest_path else None
    sidecar = _load_json(sidecar_path) if sidecar_path else None

    for source_name, source in (("run_status", status), ("manifest", manifest), ("maintenance", sidecar)):
        if not source:
            continue
        git_data = source.get("git") if isinstance(source.get("git"), dict) else {}
        commit = source.get("git_commit") or source.get("commit") or git_data.get("commit") or git_data.get("sha")
        branch = source.get("git_branch") or source.get("branch") or git_data.get("branch")
        if commit:
            return {
                "commit": str(commit),
                "branch": str(branch) if branch else None,
                "source": source_name,
                "inferred": bool(source.get("inferred")) or source_name == "maintenance",
            }
    return {"commit": None, "branch": None, "source": "unknown", "inferred": False}


def _is_ancestor(older: str, newer: str) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def classify_run_version(run_dir: Path, root: Path) -> dict[str, Any]:
    current = current_repository_version()
    run = run_repository_provenance(run_dir, root)
    current_commit = current.get("commit")
    run_commit = run.get("commit")

    if not current_commit or not run_commit:
        status = "unknown"
        is_outdated = False
    elif current_commit == run_commit:
        status = "current"
        is_outdated = False
    else:
        ancestor = _is_ancestor(str(run_commit), str(current_commit))
        reverse = _is_ancestor(str(current_commit), str(run_commit)) if ancestor is False else None
        if ancestor is True:
            status = "outdated"
            is_outdated = True
        elif reverse is True:
            status = "newer"
            is_outdated = False
        elif ancestor is False and reverse is False:
            status = "different"
            is_outdated = False
        else:
            same_branch = not run.get("branch") or not current.get("branch") or run.get("branch") == current.get("branch")
            status = "outdated" if same_branch else "different"
            is_outdated = same_branch

    return {
        "status": status,
        "is_outdated": is_outdated,
        "run_commit": run_commit,
        "run_branch": run.get("branch"),
        "run_source": run.get("source"),
        "run_inferred": run.get("inferred", False),
        "current_commit": current_commit,
        "current_branch": current.get("branch"),
        "current_source": current.get("source"),
    }


def annotate_run_summary(summary: dict[str, Any], run_dir: Path, root: Path) -> dict[str, Any]:
    summary["repository_version"] = classify_run_version(run_dir, root)
    return summary

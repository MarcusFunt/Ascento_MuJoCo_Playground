"""Capture reproducible provenance for explicitly allowed dirty-source runs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkingTreeState:
    commit: str | None
    branch: str | None
    tracked_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]

    @property
    def is_dirty(self) -> bool:
        return bool(self.tracked_paths or self.untracked_paths)


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _optional_git_value(repo_root: Path, *args: str) -> str | None:
    value = _git(repo_root, *args).strip()
    return value or None


def working_tree_state(repo_root: Path) -> WorkingTreeState:
    """Describe all nonignored source changes relative to the current commit."""
    root = repo_root.expanduser().resolve()
    tracked: list[str] = []
    untracked: list[str] = []
    for line in _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines():
        if not line or line.startswith("!!"):
            continue
        path = line[3:]
        if line.startswith("??"):
            untracked.append(path)
        else:
            tracked.append(path)
    return WorkingTreeState(
        commit=_optional_git_value(root, "rev-parse", "HEAD"),
        branch=_optional_git_value(root, "branch", "--show-current"),
        tracked_paths=tuple(sorted(tracked)),
        untracked_paths=tuple(sorted(untracked)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dirty_source_bundle(repo_root: Path, destination: Path) -> dict[str, Any]:
    """Write a patch, archive, and atomic manifest for a dirty source tree."""
    root = repo_root.expanduser().resolve()
    bundle_dir = destination.expanduser().resolve()
    state = working_tree_state(root)
    bundle_dir.mkdir(parents=True, exist_ok=False)

    patch_path = bundle_dir / "tracked.patch"
    patch_path.write_text(_git(root, "diff", "--binary", "HEAD"), encoding="utf-8")
    _git(root, "apply", "--check", "--cached", str(patch_path))

    untracked: dict[str, dict[str, Any]] = {}
    archive_path = bundle_dir / "untracked.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for relative_path in state.untracked_paths:
            path = (root / relative_path).resolve()
            if root not in path.parents:
                raise ValueError(f"untracked path escapes repository root: {relative_path}")
            if not path.is_file():
                raise ValueError(f"untracked path is not a regular file: {relative_path}")
            archive.add(path, arcname=relative_path, recursive=False)
            untracked[relative_path] = {"sha256": _sha256(path), "size": path.stat().st_size}

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "commit": state.commit,
        "branch": state.branch,
        "tracked_paths": list(state.tracked_paths),
        "untracked_paths": list(state.untracked_paths),
        "tracked_patch": {"path": patch_path.name, "sha256": _sha256(patch_path)},
        "untracked_archive": {"path": archive_path.name, "sha256": _sha256(archive_path)},
        "untracked": untracked,
    }
    manifest_path = bundle_dir / "manifest.json"
    temporary_path = manifest_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, manifest_path)
    return manifest

"""Capture reproducible provenance for explicitly allowed dirty-source runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkingTreeState:
    commit: str | None
    branch: str | None
    tracked_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]
    source: str = "git"

    @property
    def is_dirty(self) -> bool:
        return bool(self.tracked_paths or self.untracked_paths)


def _git(repo_root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=command_env,
    )
    return result.stdout


def _optional_git_value(repo_root: Path, *args: str) -> str | None:
    value = _git(repo_root, *args).strip()
    return value or None


def working_tree_state(repo_root: Path) -> WorkingTreeState:
    """Describe all nonignored source changes relative to the current commit."""
    root = repo_root.expanduser().resolve()
    if not (root / ".git").exists():
        commit = os.environ.get("ASCENTO_REPOSITORY_COMMIT") or None
        branch = os.environ.get("ASCENTO_REPOSITORY_BRANCH") or None
        build_status = os.environ.get("ASCENTO_REPOSITORY_DIRTY", "").strip().lower()
        if not commit or build_status not in {"0", "false", "clean"}:
            raise ValueError(
                "repository Git metadata is unavailable; managed runs require a known commit "
                "from a verified clean image provenance build"
            )
        return WorkingTreeState(commit, branch, (), (), source="image")

    tracked = tuple(
        sorted(
            path
            for path in _git(root, "diff", "--name-only", "--no-renames", "-z", "HEAD").split("\0")
            if path
        )
    )
    untracked = tuple(
        sorted(
            path
            for path in _git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
            if path
        )
    )
    return WorkingTreeState(
        commit=_optional_git_value(root, "rev-parse", "HEAD"),
        branch=_optional_git_value(root, "branch", "--show-current"),
        tracked_paths=tracked,
        untracked_paths=untracked,
        source="git",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        _update_digest(digest, handle)
    return digest.hexdigest()


def _update_digest(digest, handle) -> int:
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return size


def _untracked_manifest(root: Path, paths: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative_path in paths:
        path = root / relative_path
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError(f"untracked path escapes repository root: {relative_path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"untracked path is not a regular file: {relative_path}")
        records[relative_path] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return records


def _patch_paths(repo_root: Path, patch_path: Path) -> tuple[str, ...]:
    if patch_path.stat().st_size == 0:
        return ()
    numstat = _git(repo_root, "apply", "--numstat", "-z", str(patch_path))
    paths = []
    for record in numstat.split("\0"):
        if not record:
            continue
        fields = record.split("\t", maxsplit=2)
        if len(fields) != 3:
            raise ValueError("tracked source patch is incomplete or malformed")
        paths.append(fields[2])
    return tuple(sorted(paths))


def write_dirty_source_bundle(repo_root: Path, destination: Path) -> dict[str, Any]:
    """Write a patch, archive, and atomic manifest for a dirty source tree."""
    root = repo_root.expanduser().resolve()
    bundle_dir = destination.expanduser().resolve()
    state = working_tree_state(root)
    if state.source != "git" or state.commit is None:
        raise ValueError("dirty source bundles require an available Git checkout")
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists():
        raise FileExistsError(f"provenance bundle already exists: {bundle_dir}")

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{bundle_dir.name}.capture-", dir=bundle_dir.parent)
    )
    try:
        tracked_patch = _git(root, "diff", "--binary", "--no-renames", "HEAD")
        patch_paths = tuple(
            sorted(
                path
                for path in _git(
                    root, "diff", "--name-only", "--no-renames", "-z", "HEAD"
                ).split("\0")
                if path
            )
        )
        if patch_paths != state.tracked_paths:
            raise ValueError("tracked source changed while provenance capture was starting; retry")

        patch_path = staging_dir / "tracked.patch"
        patch_path.write_text(tracked_patch, encoding="utf-8")
        if _patch_paths(root, patch_path) != state.tracked_paths:
            raise ValueError("tracked source patch does not cover the tracked source snapshot")
        with tempfile.TemporaryDirectory(prefix="ascento-provenance-index-") as index_dir:
            index_path = str(Path(index_dir) / "index")
            index_env = {"GIT_INDEX_FILE": index_path}
            _git(root, "read-tree", state.commit, env=index_env)
            _git(root, "apply", "--check", "--cached", str(patch_path), env=index_env)

        untracked = _untracked_manifest(root, state.untracked_paths)
        archive_path = staging_dir / "untracked.tar"
        with tarfile.open(archive_path, mode="w") as archive:
            for relative_path in state.untracked_paths:
                archive.add(root / relative_path, arcname=relative_path, recursive=False)

        with tarfile.open(archive_path, mode="r") as archive:
            for relative_path, expected in untracked.items():
                member = archive.extractfile(relative_path)
                if member is None:
                    raise ValueError(f"untracked archive is incomplete: {relative_path}")
                digest = hashlib.sha256()
                size = _update_digest(digest, member)
                if digest.hexdigest() != expected["sha256"] or size != expected["size"]:
                    raise ValueError(f"untracked file changed during provenance capture: {relative_path}")

        state_after = working_tree_state(root)
        patch_after = _git(root, "diff", "--binary", "--no-renames", "HEAD")
        untracked_after = _untracked_manifest(root, state_after.untracked_paths)
        if state_after != state or patch_after != tracked_patch or untracked_after != untracked:
            raise ValueError("source tree changed during provenance capture; retry the run")

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
        manifest_path = staging_dir / "manifest.json"
        temporary_path = manifest_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary_path, manifest_path)
        if bundle_dir.exists():
            raise FileExistsError(f"provenance bundle already exists: {bundle_dir}")
        os.rename(staging_dir, bundle_dir)
        return manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

"""Collect reproducibility metadata for a significant Ascento run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _model_hash(repo_root: Path, model_path: Path | None) -> str | None:
    candidate = model_path or Path(
        os.environ.get("ASCENTO_MODEL_PATH", "model/ascento_guard2_mjx.xml")
    )
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(
    repo_root: Path,
    *,
    stage: str | None = None,
    training_arguments: dict[str, Any] | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Return environment, source, model, and run configuration metadata."""
    git_commit = _git_value(repo_root, "rev-parse", "HEAD")
    dirty = _git_value(repo_root, "status", "--porcelain")
    driver = _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    try:
        import jax

        devices = [str(device) for device in jax.devices()]
    except ImportError:
        devices = []

    return {
        "git_commit": git_commit or os.environ.get("ASCENTO_BUILD_GIT_COMMIT", "unknown"),
        "git_dirty": bool(dirty),
        "docker_image": os.environ.get("ASCENTO_DOCKER_IMAGE", "unknown"),
        "source_mode": os.environ.get("ASCENTO_SOURCE_MODE", "development"),
        "python_version": platform.python_version(),
        "jax_version": _package_version("jax"),
        "mujoco_version": _package_version("mujoco"),
        "brax_version": _package_version("brax"),
        "nvidia_driver": driver,
        "jax_devices": devices,
        "physics_profile": os.environ.get("ASCENTO_PHYSICS_PROFILE", "ascento_guard2_mjx"),
        "model_hash": _model_hash(repo_root, model_path),
        "training_stage": stage,
        "training_arguments": training_arguments or {},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host_platform": platform.platform(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    record = collect_provenance(
        args.repo_root.resolve(),
        stage=args.stage,
        model_path=args.model_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

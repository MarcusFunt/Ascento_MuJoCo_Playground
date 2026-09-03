"""Shared dashboard configuration and startup validation."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "logs" / "rsl_rl"
DEFAULT_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
DEFAULT_STALE_AFTER_SECONDS = 90.0


@dataclass(frozen=True)
class DashboardConfig:
    repo_root: Path
    artifact_root: Path
    frontend_dist: Path
    stale_after_seconds: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "artifact_root": str(self.artifact_root),
            "frontend_dist": str(self.frontend_dist),
            "frontend_built": (self.frontend_dist / "index.html").is_file(),
            "stale_after_seconds": self.stale_after_seconds,
            "python_executable": sys.executable,
            "uv_available": shutil.which("uv") is not None,
            "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        }


def _positive_float(name: str, raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero, got {value}")
    return value


def load_config() -> DashboardConfig:
    artifact_root = Path(
        os.environ.get("ASCENTO_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))
    ).expanduser().resolve()
    frontend_dist = Path(
        os.environ.get("ASCENTO_DASHBOARD_DIST", str(DEFAULT_FRONTEND_DIST))
    ).expanduser().resolve()
    stale_after = _positive_float(
        "ASCENTO_STALE_AFTER_SECONDS",
        os.environ.get("ASCENTO_STALE_AFTER_SECONDS"),
        DEFAULT_STALE_AFTER_SECONDS,
    )
    return DashboardConfig(
        repo_root=REPO_ROOT,
        artifact_root=artifact_root,
        frontend_dist=frontend_dist,
        stale_after_seconds=stale_after,
    )


def validate_startup(config: DashboardConfig, *, create_artifact_root: bool = True) -> list[str]:
    """Validate paths early and return non-fatal startup warnings.

    The dashboard server is a read-only monitor and should pass
    ``create_artifact_root=False``. Writers such as the launcher may retain the
    default and create the configured artifact root before starting a run.
    """
    warnings: list[str] = []

    if config.artifact_root.exists() and not config.artifact_root.is_dir():
        raise RuntimeError(
            "Dashboard artifact root is not a directory: "
            f"{config.artifact_root}. Set ASCENTO_ARTIFACT_ROOT to a directory."
        )

    if create_artifact_root:
        try:
            config.artifact_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise RuntimeError(
                "Dashboard could not create the artifact root "
                f"{config.artifact_root}: {error}. "
                "Set ASCENTO_ARTIFACT_ROOT to a writable directory."
            ) from error
    elif not config.artifact_root.exists():
        warnings.append(
            "Dashboard artifact root does not exist yet: "
            f"{config.artifact_root}. No runs will be shown until training creates it."
        )

    if config.artifact_root.exists():
        if not os.access(config.artifact_root, os.R_OK):
            raise RuntimeError(f"Dashboard artifact root is not readable: {config.artifact_root}")
        if not os.access(config.artifact_root, os.X_OK):
            raise RuntimeError(f"Dashboard artifact root is not searchable: {config.artifact_root}")

    if config.frontend_dist.exists() and not config.frontend_dist.is_dir():
        raise RuntimeError(
            "ASCENTO_DASHBOARD_DIST must point to a directory, got "
            f"{config.frontend_dist}"
        )
    if not (config.frontend_dist / "index.html").is_file():
        warnings.append(
            "Dashboard frontend is not built. Run "
            "`cd dashboard/frontend && npm install && npm run build`."
        )

    return warnings

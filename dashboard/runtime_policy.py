"""Enforce the one supported training source and record where code actually ran."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping

CANONICAL_CHECKOUT = Path("/root/Ascento_MuJoCo_Playground")


class RuntimePolicyError(ValueError):
    """A managed launch is not using the canonical source or requested device."""


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimePolicyError(f"cannot verify canonical checkout: {error}") from error
    return result.stdout.strip()


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def training_device(arguments: list[str]) -> str:
    """Get an explicit trainer device, defaulting managed training to the GPU."""
    names = ("--device", "--env.device", "--agent.device")
    for index, argument in enumerate(arguments):
        if argument in names and index + 1 < len(arguments):
            return arguments[index + 1]
        for name in names:
            prefix = name + "="
            if argument.startswith(prefix):
                return argument[len(prefix) :]
    return "cuda:0"


def apply_training_device(arguments: list[str], device: str) -> list[str]:
    """Return trainer arguments with one explicit resolved device option."""
    updated = list(arguments)
    names = ("--device", "--env.device", "--agent.device")
    for index, argument in enumerate(updated):
        if argument in names and index + 1 < len(updated):
            updated[index + 1] = device
            return updated
        for name in names:
            if argument.startswith(name + "="):
                updated[index] = f"{name}={device}"
                return updated
    updated.extend(["--device", device])
    return updated


def _normalized_device(value: str, probe: Callable[[], bool]) -> str:
    if value == "auto":
        value = "cuda:0" if probe() else "cpu"
    if value not in {"cpu", "cuda"} and re.fullmatch(r"cuda:\d+", value) is None:
        raise RuntimePolicyError(
            f"unsupported compute device {value!r}; choose cpu, cuda, cuda:N, or auto"
        )
    return value


def _identity(
    *,
    runtime_kind: str,
    source_root: Path,
    execution_root: Path,
    commit: str,
    branch: str,
    dirty: bool,
    device: str,
    compute_backend: str,
) -> dict[str, object]:
    return {
        "runtime_kind": runtime_kind,
        "source_checkout_root": str(source_root.resolve()),
        "execution_root": str(execution_root.resolve()),
        "source_commit": commit,
        "source_branch": branch,
        "source_dirty": dirty,
        "compute_backend": compute_backend,
        "device": device,
        "python_executable": sys.executable,
    }


def runtime_identity(
    execution_root: Path,
    *,
    canonical_root: Path = CANONICAL_CHECKOUT,
    env: Mapping[str, str] | None = None,
    requested_device: str = "cuda:0",
    cuda_probe: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Validate a native checkout or immutable Docker build and describe it.

    Native training is accepted only from a clean checkout at origin/main.
    Docker uses source and Git metadata mounted read-only from that same WSL
    checkout; its built dependencies and frontend must match the mounted revision.
    """
    variables = os.environ if env is None else env
    probe = cuda_probe or cuda_available
    probe_is_injected = cuda_probe is not None
    execution = execution_root.expanduser().resolve()
    canonical = canonical_root.expanduser().resolve()
    device = _normalized_device(requested_device, probe)
    kind = variables.get("ASCENTO_RUNTIME_KIND", "checkout")

    if kind == "packaged-docker":
        source_value = variables.get("ASCENTO_CANONICAL_SOURCE_ROOT", "")
        version_value = variables.get("ASCENTO_CANONICAL_VERSION_FILE", "")
        if not source_value or Path(source_value).expanduser().resolve() != canonical:
            raise RuntimePolicyError(
                f"Docker image is not pinned to the canonical WSL checkout {canonical}"
            )
        if execution != Path("/workspace"):
            raise RuntimePolicyError(
                f"packaged Docker runtime must execute from /workspace, got {execution}"
            )
        if not version_value:
            raise RuntimePolicyError("Docker runtime is missing its canonical revision manifest")
        try:
            manifest = json.loads(Path(version_value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimePolicyError(
                f"cannot read canonical repository-version manifest: {error}"
            ) from error
        commit = variables.get("ASCENTO_REPOSITORY_COMMIT", "")
        branch = variables.get("ASCENTO_REPOSITORY_BRANCH", "")
        dirty_value = variables.get("ASCENTO_REPOSITORY_DIRTY", "").lower()
        compute = variables.get("ASCENTO_COMPUTE_EXTRA", "")
        if (
            not commit
            or branch != "main"
            or dirty_value != "0"
            or manifest.get("commit") != commit
            or manifest.get("branch") != branch
            or manifest.get("compute") != compute
        ):
            raise RuntimePolicyError(
                "Docker image revision does not match canonical checkout manifest; "
                "rebuild the image from clean origin/main"
            )
        git_root = Path(_git(execution, "rev-parse", "--show-toplevel")).resolve()
        live_branch = _git(execution, "branch", "--show-current")
        live_commit = _git(execution, "rev-parse", "HEAD")
        remote_commit = _git(execution, "rev-parse", "origin/main")
        dirty_status = _git(execution, "status", "--porcelain", "--untracked-files=all")
        if (
            git_root != execution
            or live_branch != "main"
            or live_commit != commit
            or remote_commit != commit
            or dirty_status
        ):
            raise RuntimePolicyError(
                "Docker runtime source does not match the clean canonical origin/main checkout; "
                "rebuild or synchronize the runtime before starting a run"
            )
        result = _identity(
            runtime_kind=kind,
            source_root=canonical,
            execution_root=execution,
            commit=commit,
            branch=branch,
            dirty=False,
            device=device,
            compute_backend=compute,
        )
    else:
        if kind not in {"checkout", "canonical-wsl-checkout"}:
            raise RuntimePolicyError(f"unsupported training runtime kind: {kind}")
        if execution != canonical:
            raise RuntimePolicyError(
                f"managed training must use the canonical WSL checkout {canonical}; "
                f"this process is running from {execution}"
            )
        if not sys.platform.startswith("linux"):
            raise RuntimePolicyError("managed training must run in Linux/WSL, not Windows")
        git_root = Path(_git(execution, "rev-parse", "--show-toplevel")).resolve()
        branch = _git(execution, "branch", "--show-current")
        commit = _git(execution, "rev-parse", "HEAD")
        remote_commit = _git(execution, "rev-parse", "origin/main")
        dirty_status = _git(execution, "status", "--porcelain", "--untracked-files=all")
        if git_root != canonical:
            raise RuntimePolicyError(
                f"managed training resolved to {git_root}, not canonical checkout {canonical}"
            )
        if branch != "main":
            raise RuntimePolicyError(f"managed training requires branch main, found {branch or 'detached'}")
        if dirty_status:
            raise RuntimePolicyError(
                "managed training requires a clean canonical checkout; commit or preserve local source changes first"
            )
        if commit != remote_commit:
            raise RuntimePolicyError(
                f"canonical checkout is not at origin/main ({commit[:12]} != {remote_commit[:12]}); "
                "synchronize main before launching"
            )
        result = _identity(
            runtime_kind="canonical-wsl-checkout",
            source_root=canonical,
            execution_root=execution,
            commit=commit,
            branch=branch,
            dirty=False,
            device=device,
            compute_backend="cu128" if device.startswith("cuda") else "cpu",
        )

    if device.startswith("cuda"):
        if not probe():
            raise RuntimePolicyError(
                "CUDA is unavailable in the selected Python runtime; use the cu128 GPU environment "
                "or explicitly request --device cpu"
            )
        if probe_is_injected:
            result["gpu_name"] = None
            result["torch_cuda_version"] = None
        else:
            try:
                import torch

                result["gpu_name"] = torch.cuda.get_device_name(device)
                result["torch_cuda_version"] = torch.version.cuda
            except (ImportError, RuntimeError, ValueError, TypeError, AssertionError) as error:
                raise RuntimePolicyError(
                    f"CUDA device {device} is unavailable in the selected Python runtime: {error}"
                ) from error
    return result

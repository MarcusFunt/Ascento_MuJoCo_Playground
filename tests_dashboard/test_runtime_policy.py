import json
from pathlib import Path

import pytest
from dashboard.runtime_policy import RuntimePolicyError, runtime_identity, training_device


def _host_git(monkeypatch, root: Path, *, commit="abc123", remote="abc123", dirty=""):
    values = {
        ("rev-parse", "--show-toplevel"): str(root),
        ("branch", "--show-current"): "main",
        ("rev-parse", "HEAD"): commit,
        ("rev-parse", "origin/main"): remote,
        ("status", "--porcelain", "--untracked-files=all"): dirty,
    }

    def fake_git(_root, *args):
        return values[args]

    monkeypatch.setattr("dashboard.runtime_policy._git", fake_git)


def test_native_runtime_rejects_a_second_checkout(tmp_path):
    canonical = tmp_path / "canonical"
    other = tmp_path / "other"

    with pytest.raises(RuntimePolicyError, match="canonical WSL checkout"):
        runtime_identity(other, canonical_root=canonical, env={})


def test_native_runtime_requires_clean_main_at_origin_head(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _host_git(monkeypatch, canonical, commit="old", remote="new")

    with pytest.raises(RuntimePolicyError, match="origin/main"):
        runtime_identity(canonical, canonical_root=canonical, env={})


def test_native_runtime_records_the_actual_source_and_python(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _host_git(monkeypatch, canonical)

    value = runtime_identity(
        canonical,
        canonical_root=canonical,
        env={},
        requested_device="cuda:0",
        cuda_probe=lambda: True,
    )

    assert value["runtime_kind"] == "canonical-wsl-checkout"
    assert value["source_checkout_root"] == str(canonical.resolve())
    assert value["execution_root"] == str(canonical.resolve())
    assert value["source_commit"] == "abc123"
    assert value["source_branch"] == "main"
    assert value["device"] == "cuda:0"
    assert value["python_executable"]


def test_native_runtime_fails_before_a_cuda_launch_without_cuda(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _host_git(monkeypatch, canonical)

    with pytest.raises(RuntimePolicyError, match="CUDA is unavailable"):
        runtime_identity(
            canonical,
            canonical_root=canonical,
            env={},
            requested_device="cuda:0",
            cuda_probe=lambda: False,
        )


def test_runtime_rejects_unrecognized_compute_device(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _host_git(monkeypatch, canonical)

    with pytest.raises(RuntimePolicyError, match="unsupported compute device"):
        runtime_identity(
            canonical,
            canonical_root=canonical,
            env={},
            requested_device="mps",
            cuda_probe=lambda: True,
        )


def test_docker_runtime_requires_image_revision_to_match_canonical_manifest(monkeypatch, tmp_path):
    canonical = tmp_path / "canonical"
    version_file = tmp_path / "repository-version.json"
    version_file.write_text(
        json.dumps({"commit": "abc123", "branch": "main", "compute": "cu128"}),
        encoding="utf-8",
    )
    _host_git(monkeypatch, Path("/workspace"))
    env = {
        "ASCENTO_RUNTIME_KIND": "packaged-docker",
        "ASCENTO_CANONICAL_SOURCE_ROOT": str(canonical),
        "ASCENTO_CANONICAL_VERSION_FILE": str(version_file),
        "ASCENTO_REPOSITORY_COMMIT": "abc123",
        "ASCENTO_REPOSITORY_BRANCH": "main",
        "ASCENTO_REPOSITORY_DIRTY": "0",
        "ASCENTO_COMPUTE_EXTRA": "cu128",
    }

    value = runtime_identity(
        Path("/workspace"),
        canonical_root=canonical,
        env=env,
        requested_device="cuda:0",
        cuda_probe=lambda: True,
    )

    assert value["runtime_kind"] == "packaged-docker"
    assert value["source_checkout_root"] == str(canonical.resolve())
    assert value["execution_root"] == "/workspace"
    assert value["source_commit"] == "abc123"

    env["ASCENTO_REPOSITORY_COMMIT"] = "stale"
    with pytest.raises(RuntimePolicyError, match="does not match canonical checkout"):
        runtime_identity(
            Path("/workspace"),
            canonical_root=canonical,
            env=env,
            requested_device="cpu",
        )


def test_training_device_defaults_to_cuda_and_honors_explicit_cpu():
    assert training_device([]) == "cuda:0"
    assert training_device(["--device", "cpu"]) == "cpu"
    assert training_device(["--device=cuda:1"]) == "cuda:1"
    assert training_device(["--agent.device", "cpu"]) == "cpu"

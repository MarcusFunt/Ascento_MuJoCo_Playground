from pathlib import Path

import pytest

from dashboard.config import load_config, validate_startup
from dashboard.launch import _training_arg, build_parser


def test_launcher_uses_same_default_artifact_root_as_dashboard(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCENTO_ARTIFACT_ROOT", str(tmp_path))
    config = load_config()
    args = build_parser().parse_args([])

    assert args.artifact_root == config.artifact_root == tmp_path.resolve()


def test_launcher_argument_metadata_parser_supports_both_cli_forms():
    args = [
        "--seed=7",
        "--device",
        "cuda:0",
        "--env.sim.dt",
        "0.002",
        "--agent.seed",
        "11",
    ]

    assert _training_arg(args, "--seed", "--agent.seed") == "11"
    assert _training_arg(args, "--device") == "cuda:0"
    assert _training_arg(args, "--env.sim.dt") == "0.002"


def test_startup_validation_reports_bad_artifact_root(tmp_path):
    bad_root = tmp_path / "artifact-file"
    bad_root.write_text("not a directory", encoding="utf-8")
    monkeypatch_config = load_config()
    config = type(monkeypatch_config)(
        repo_root=monkeypatch_config.repo_root,
        artifact_root=bad_root,
        frontend_dist=monkeypatch_config.frontend_dist,
        stale_after_seconds=monkeypatch_config.stale_after_seconds,
    )

    with pytest.raises(RuntimeError, match="artifact root is not a directory"):
        validate_startup(config)

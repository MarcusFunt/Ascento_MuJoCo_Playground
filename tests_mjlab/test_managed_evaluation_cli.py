from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ascento_mjlab.cli as cli
import ascento_mjlab.evaluation.cli as evaluation_cli


def test_managed_evaluation_cli_accepts_and_passes_obstacle_mode_zero(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    captured: dict[str, object] = {}
    args = cli.build_parser().parse_args(
        [
            "evaluate",
            "run",
            "--checkpoint",
            str(checkpoint),
            "--suite",
            "generalist_reward_alignment_v1",
            "--device",
            "cpu",
            "--obstacle-mode",
            "0",
        ]
    )

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(value="PASS"), tmp_path

    monkeypatch.setattr(cli, "_checkpoint_from_args", lambda _args: checkpoint)
    monkeypatch.setattr(cli, "evaluation_root", lambda _args: tmp_path)
    monkeypatch.setattr(cli, "evaluation_details", lambda _directory, _output: {})
    monkeypatch.setattr(cli, "_print", lambda _payload, _json: None)
    monkeypatch.setattr(evaluation_cli, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluation_cli, "resolve_suite_path", lambda suite: suite)

    args.handler(args)

    assert captured["obstacle_mode"] == 0

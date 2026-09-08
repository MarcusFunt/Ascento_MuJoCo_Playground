import json
import subprocess
import sys
import zipfile

import pytest

from ascento_mjlab.operations import (
    archive_evaluation,
    evaluation_details,
    latest_checkpoint,
    list_evaluations,
    resolve_evaluation_dir,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluation(root):
    directory = root / "example"
    directory.mkdir(parents=True)
    _write_json(
        directory / "manifest.json",
        {
            "suite_id": "balance_gate_v2",
            "task": "Ascento-Balance-Flat",
            "checkpoint": "/models/model_100.pt",
            "checkpoint_sha256": "abc",
            "repository_commit": "deadbeef",
            "repository_dirty": False,
            "scenario_count": 12,
            "started_at_utc": "2026-09-08T10:00:00Z",
            "finished_at_utc": "2026-09-08T10:01:00Z",
        },
    )
    _write_json(directory / "gate.json", {"status": "PASS", "gates": []})
    _write_json(directory / "summary.json", {"nominal": {"success": {"p95": 1.0}}})
    _write_json(directory / "consistency.json", {"passed": True, "checks": []})
    _write_json(directory / "failures.json", {"worst": []})
    _write_json(directory / "clips_manifest.json", {"status": "ok", "captures": []})
    (directory / "report.html").write_text("<html>report</html>", encoding="utf-8")
    return directory


def test_evaluation_artifacts_can_be_listed_inspected_and_archived(tmp_path):
    root = tmp_path / "evaluations"
    directory = _evaluation(root)

    reports = list_evaluations(root)
    assert reports[0]["id"] == "example"
    assert reports[0]["status"] == "PASS"

    details = evaluation_details("example", root)
    assert details["summary"]["nominal"]["success"]["p95"] == 1.0
    assert details["clips_manifest"]["status"] == "ok"

    archive = archive_evaluation("example", root=root)
    assert archive.is_file()
    with zipfile.ZipFile(archive) as handle:
        assert sorted(handle.namelist()) == sorted(
            path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
        )


def test_evaluation_artifact_operations_reject_paths_outside_the_output_root(tmp_path):
    root = tmp_path / "evaluations"
    _evaluation(root)

    with pytest.raises(ValueError, match="below"):
        resolve_evaluation_dir(tmp_path, root)


def test_partial_evaluations_are_visible_instead_of_silently_omitted(tmp_path):
    root = tmp_path / "evaluations"
    partial = root / "interrupted"
    partial.mkdir(parents=True)
    _write_json(
        partial / "suite.json",
        {"suite_id": "balance_gate_v2", "task": "Ascento-Balance-Flat"},
    )

    reports = list_evaluations(root)
    assert reports[0]["id"] == "interrupted"
    assert reports[0]["status"] == "INCOMPLETE"
    assert "manifest.json" in reports[0]["incomplete_reason"]

    details = evaluation_details("interrupted", root)
    assert details["suite"]["suite_id"] == "balance_gate_v2"


def test_latest_checkpoint_prefers_the_highest_numbered_checkpoint(tmp_path):
    (tmp_path / "model_100.pt").write_bytes(b"old")
    nested = tmp_path / "checkpoint"
    nested.mkdir()
    newest = nested / "model_250.pt"
    newest.write_bytes(b"new")
    (tmp_path / "model_best_long_horizon.pt").write_bytes(b"candidate")

    assert latest_checkpoint(tmp_path) == newest.resolve()


def test_cli_and_mcp_import_dashboard_from_outside_the_checkout(tmp_path):
    """Installed entry points must not depend on the caller's current directory."""
    for source in (
        "import ascento_mjlab.cli; import dashboard; print(dashboard.__name__)",
        "import ascento_mjlab.mcp_server; import dashboard; print(dashboard.__name__)",
    ):
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

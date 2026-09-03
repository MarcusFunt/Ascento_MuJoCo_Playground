import json

from dashboard.versioning import (
    classify_run_version,
    current_repository_version,
    run_repository_provenance,
)
from scripts.stamp_run_provenance import stamp_missing_runs


def _set_current(monkeypatch, commit="newcommit", branch="main"):
    monkeypatch.setenv("ASCENTO_REPOSITORY_COMMIT", commit)
    monkeypatch.setenv("ASCENTO_REPOSITORY_BRANCH", branch)
    current_repository_version.cache_clear()


def test_same_commit_is_current(monkeypatch, tmp_path):
    _set_current(monkeypatch, "abc123", "main")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_status.json").write_text(
        json.dumps({"git_commit": "abc123", "git_branch": "main"}),
        encoding="utf-8",
    )

    result = classify_run_version(run, tmp_path)
    assert result["status"] == "current"
    assert result["is_outdated"] is False


def test_same_branch_mismatch_is_flagged_outdated(monkeypatch, tmp_path):
    _set_current(monkeypatch, "newcommit", "main")
    run = tmp_path / "run"
    run.mkdir()
    (run / "repository_provenance.json").write_text(
        json.dumps({"commit": "oldcommit", "branch": "main", "inferred": True}),
        encoding="utf-8",
    )

    result = classify_run_version(run, tmp_path)
    assert result["status"] == "outdated"
    assert result["is_outdated"] is True
    assert result["run_inferred"] is True


def test_exact_launcher_metadata_wins_over_maintenance_sidecar(monkeypatch, tmp_path):
    _set_current(monkeypatch, "current", "main")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_status.json").write_text(
        json.dumps({"git_commit": "exact", "git_branch": "feature"}),
        encoding="utf-8",
    )
    (run / "repository_provenance.json").write_text(
        json.dumps({"commit": "inferred", "branch": "main", "inferred": True}),
        encoding="utf-8",
    )

    provenance = run_repository_provenance(run, tmp_path)
    assert provenance["commit"] == "exact"
    assert provenance["source"] == "run_status"
    assert provenance["inferred"] is False


def test_maintenance_stamps_only_legacy_runs_without_git_metadata(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "telemetry.jsonl").write_text('{"completed_steps": 1}\n', encoding="utf-8")

    exact = tmp_path / "exact"
    exact.mkdir()
    (exact / "telemetry.jsonl").write_text('{"completed_steps": 1}\n', encoding="utf-8")
    (exact / "run_status.json").write_text(
        json.dumps({"git_commit": "exactcommit", "git_branch": "feature"}),
        encoding="utf-8",
    )

    assert stamp_missing_runs(tmp_path, "oldcheckout", "main") == 1
    inferred = json.loads((legacy / "repository_provenance.json").read_text(encoding="utf-8"))
    assert inferred["commit"] == "oldcheckout"
    assert inferred["branch"] == "main"
    assert inferred["inferred"] is True
    assert not (exact / "repository_provenance.json").exists()

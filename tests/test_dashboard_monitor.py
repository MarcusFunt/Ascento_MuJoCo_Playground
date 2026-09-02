import json
from pathlib import Path

from dashboard.monitor import discover_runs, error_excerpt, load_jsonl, summarize_run


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_discover_and_summarize_run(tmp_path):
    run = tmp_path / "20260902_balance"
    write_jsonl(
        run / "telemetry.jsonl",
        [
            {
                "stage": "balance",
                "completed_steps": 250,
                "total_steps": 1000,
                "percent_complete": 25.0,
                "steps_per_second": 100.0,
                "eta_seconds": 7.5,
                "metrics": {"eval/episode_reward": 3.0},
            }
        ],
    )

    refs = discover_runs(tmp_path)
    assert len(refs) == 1
    summary = summarize_run(run, tmp_path, now=(run / "telemetry.jsonl").stat().st_mtime + 1)
    assert summary["stage"] == "balance"
    assert summary["state"] == "running"
    assert summary["telemetry"]["percent_complete"] == 25.0


def test_manifest_marks_run_finished(tmp_path):
    run = tmp_path / "finished"
    run.mkdir()
    (run / "training_manifest.json").write_text(
        json.dumps({"stage": {"name": "recovery"}}), encoding="utf-8"
    )
    summary = summarize_run(run, tmp_path)
    assert summary["state"] == "finished"
    assert summary["stage"] == "recovery"


def test_status_stage_is_available_before_first_telemetry(tmp_path):
    run = tmp_path / "startup_run"
    run.mkdir()
    (run / "run_status.json").write_text(
        json.dumps({"state": "running", "stage": "balance"}), encoding="utf-8"
    )

    summary = summarize_run(run, tmp_path)

    assert summary["stage"] == "balance"


def test_explicit_error_status_wins(tmp_path):
    run = tmp_path / "failed"
    run.mkdir()
    (run / "run_status.json").write_text(
        json.dumps({"state": "error", "exit_code": 1}), encoding="utf-8"
    )
    summary = summarize_run(run, tmp_path)
    assert summary["state"] == "error"
    assert summary["status"]["exit_code"] == 1


def test_jsonl_skips_partial_or_corrupt_lines(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text('{"step": 1}\nnot-json\n{"step": 2', encoding="utf-8")
    assert load_jsonl(path) == [{"step": 1}]


def test_error_excerpt_keeps_context(tmp_path):
    log = tmp_path / "training.log"
    log.write_text(
        "normal line\n"
        "before failure\n"
        "Traceback (most recent call last):\n"
        "  File train.py, line 10\n"
        "FloatingPointError: Non-finite PPO metrics detected\n",
        encoding="utf-8",
    )
    excerpt = error_excerpt(log)
    assert any("Traceback" in line for line in excerpt)
    assert any("Non-finite" in line for line in excerpt)


def test_latest_render_is_confined_to_run(tmp_path):
    run = tmp_path / "rendered"
    renders = run / "renders"
    renders.mkdir(parents=True)
    image = renders / "balance_step_000000100.png"
    image.write_bytes(b"png")
    write_jsonl(
        renders / "progress_renders.jsonl",
        [{"step": 100, "path": str(image), "return": 4.2}],
    )
    summary = summarize_run(run, tmp_path)
    assert summary["latest_render"]["step"] == 100
    assert summary["latest_render"]["relative_path"].startswith("renders/")

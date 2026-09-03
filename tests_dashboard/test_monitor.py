import json
import os

from dashboard.health import (
    decorate_records,
    discover_dashboard_runs,
    load_dashboard_records,
    summarize_dashboard_run,
)


def test_canonical_metrics_and_non_finite_detection(tmp_path):
    record = decorate_records(
        [{
            "completed_steps": 3,
            "metrics": {
                "Train/mean_reward": 2.5,
                "Train/mean_episode_length": 42.0,
                "Loss/surrogate": 0.12,
                "Loss/entropy": -0.02,
                "Loss/kl": 0.006,
                "Policy/clip_fraction": 0.17,
                "bad_metric": float("inf"),
            },
        }],
        tmp_path,
    )[0]

    assert record["canonical_metrics"]["reward"] == 2.5
    assert record["canonical_metrics"]["episode_length"] == 42.0
    assert record["canonical_metrics"]["ppo_loss"] == 0.12
    assert record["canonical_metrics"]["kl"] == 0.006
    assert record["canonical_metrics"]["clip_fraction"] == 0.17
    assert record["metrics"]["bad_metric"] is None
    assert record["has_non_finite"] is True
    assert record["canonical_metrics"]["invalid_update"] == 1


def test_nested_launcher_run_is_discovered_once_and_inherits_metadata(tmp_path):
    launch_dir = tmp_path / "20260903_balance"
    actual_run = launch_dir / "ascento_balance" / "2026-09-03_13-00-00"
    actual_run.mkdir(parents=True)
    (launch_dir / "training.log").write_text("training\n", encoding="utf-8")
    (launch_dir / "run_status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "task": "Ascento-Balance-Flat",
                "stage": "balance",
                "seed": 12,
                "device": "cuda:0",
                "started_at": "2026-09-03T11:00:00+00:00",
                "command": ["python", "-m", "mjlab.scripts.train"],
                "pid": os.getpid(),
                "git_commit": "abc123",
                "git_branch": "feature/test",
            }
        ),
        encoding="utf-8",
    )
    params = actual_run / "params"
    params.mkdir()
    (params / "agent.yaml").write_text("max_iterations: 100\n", encoding="utf-8")
    (params / "env.yaml").write_text("sim:\n  dt: 0.002\n", encoding="utf-8")
    (actual_run / "model_20.pt").write_bytes(b"")
    (actual_run / "telemetry.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "completed_steps": 10,
                        "total_steps": 100,
                        "wall_time": 800.0,
                        "metrics": {
                            "Train/mean_reward": 1.0,
                            "Train/mean_episode_length": 30.0,
                            "Perf/total_fps": 5000.0,
                        },
                    }
                ),
                json.dumps(
                    {
                        "completed_steps": 20,
                        "total_steps": 100,
                        "wall_time": 810.0,
                        "metrics": {
                            "Train/mean_reward": 2.0,
                            "Train/mean_episode_length": 40.0,
                            "Perf/total_fps": 5200.0,
                            "training/invalid_update": 1,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    refs = discover_dashboard_runs(tmp_path)
    assert len(refs) == 1
    assert refs[0].path == actual_run.resolve()

    summary = summarize_dashboard_run(
        actual_run,
        tmp_path,
        now=1000.0,
        stale_after_seconds=90.0,
        detailed=True,
    )

    assert summary["stale"] is True
    assert summary["freshness_seconds"] == 190.0
    assert summary["process"]["alive"] is True
    assert summary["run_info"]["git_commit"] == "abc123"
    assert summary["run_info"]["task"] == "Ascento-Balance-Flat"
    assert summary["run_info"]["seed"] == 12
    assert summary["run_info"]["simulation_timestep"] == 0.002
    assert summary["run_info"]["checkpoint_path"] == "model_20.pt"
    assert summary["training_health"]["invalid_updates"] == 1
    assert summary["telemetry"]["iteration"] == 20
    assert summary["telemetry"]["eta_seconds"] == 80.0


def test_jsonl_eta_uses_iteration_rate_not_environment_fps(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "telemetry.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "completed_steps": 0,
                        "total_steps": 100,
                        "wall_time": 100.0,
                        "metrics": {"Perf/total_fps": 100000.0},
                    }
                ),
                json.dumps(
                    {
                        "completed_steps": 10,
                        "total_steps": 100,
                        "wall_time": 110.0,
                        "metrics": {"Perf/total_fps": 100000.0},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_dashboard_records(run_dir)
    assert records[-1]["steps_per_second"] == 100000.0
    assert records[-1]["iterations_per_second"] == 1.0
    assert records[-1]["eta_seconds"] == 90.0

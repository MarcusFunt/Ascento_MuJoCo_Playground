from concurrent.futures import ThreadPoolExecutor

from dashboard.config import REPO_ROOT
from dashboard.database import CheckpointIndex, DashboardDatabase, RunIndex
from sqlalchemy import select


def _database(tmp_path):
    path = (tmp_path / "dashboard.db").as_posix()
    database = DashboardDatabase(f"sqlite:///{path}", REPO_ROOT)
    database.initialize(attempts=1, retry_delay_s=0)
    assert database.available is True, database.error
    return database


def test_database_migrates_and_records_curriculum_transitions(tmp_path):
    database = _database(tmp_path)
    row = {
        "id": "run-1",
        "display_name": "Balance",
        "name": "artifact/run-1",
        "task": "Ascento-Balance-Flat",
        "stage": "balance",
        "state": "running",
        "iteration": 100,
        "repository_version": {"status": "current", "run_commit": "abc"},
    }

    database.sync_run(row, {"kind": "horizon", "stage": 1})
    # A lightweight list refresh must not erase the last curriculum snapshot.
    database.sync_run({**row, "iteration": 101})
    database.sync_run({**row, "iteration": 102}, {"kind": "horizon", "stage": 2})

    events = database.recent_events(run_id="run-1")
    transitions = [event for event in events if event["type"] == "curriculum_transition"]
    assert len(transitions) == 1
    assert transitions[0]["payload"] == {"from": 1, "to": 2}
    database.dispose()


def test_database_lists_compact_run_snapshots_for_fast_startup(tmp_path):
    database = _database(tmp_path)
    database.sync_run(
        {
            "id": "run-cached",
            "display_name": "Cached run",
            "name": "artifact/run-cached",
            "task": "Ascento-Balance-Flat",
            "stage": "balance",
            "state": "running",
            "iteration": 12,
            "total_iterations": 100,
            "percent_complete": 12.0,
            "modified_at": 1000.0,
            "repository_version": {"status": "current", "run_commit": "abc"},
        },
        {"kind": "horizon", "stage": 2},
    )

    rows = database.list_runs()

    assert database.run_artifact("run-cached") == "artifact/run-cached"
    assert len(rows) == 1
    assert rows[0]["id"] == "run-cached"
    assert rows[0]["iteration"] == 12
    assert rows[0]["repository_version"]["run_commit"] == "abc"
    assert rows[0]["curriculum"] == {"kind": "horizon", "stage": 2}
    database.dispose()


def test_database_records_run_state_transition(tmp_path):
    database = _database(tmp_path)
    row = {
        "id": "run-2",
        "display_name": "Locomotion",
        "name": "artifact/run-2",
        "state": "running",
        "repository_version": {},
    }
    database.sync_run(row)
    database.sync_run({**row, "state": "finished"})

    events = database.recent_events(run_id="run-2")
    states = [event for event in events if event["type"] == "run_state"]
    assert len(states) == 1
    assert states[0]["payload"] == {"from": "running", "to": "finished"}
    database.dispose()


def test_disabled_database_is_a_noop(tmp_path):
    database = DashboardDatabase(None, REPO_ROOT)

    database.initialize()
    database.sync_run({"id": "run"})
    database.record_event("test", "message")
    database.sync_checkpoints("run", [])
    assert database.recent_events() == []
    assert database.status() == {
        "enabled": False,
        "available": False,
        "backend": None,
        "error": None,
        "source_conflicts": None,
        "last_successful_sync_at": None,
    }


def test_concurrent_run_sync_is_idempotent(tmp_path):
    database = _database(tmp_path)
    row = {
        "id": "concurrent-run",
        "display_name": "Concurrent run",
        "name": "artifact/concurrent-run",
        "state": "running",
        "repository_version": {},
    }

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: database.sync_run(row), range(32)))

    assert database.available is True, database.error
    with database._session() as session:
        records = session.scalars(
            select(RunIndex).where(RunIndex.id == "concurrent-run")
        ).all()
    assert len(records) == 1
    assert database.recent_events(run_id="concurrent-run") == []
    database.dispose()


def test_concurrent_checkpoint_sync_is_idempotent(tmp_path):
    database = _database(tmp_path)
    checkpoint = {"relative_path": "model_100.pt", "iteration": 100, "stable": True}

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: database.sync_checkpoints("run-1", [checkpoint]), range(32)))

    assert database.available is True, database.error
    with database._session() as session:
        records = session.scalars(
            select(CheckpointIndex).where(CheckpointIndex.id == "run-1:model_100.pt")
        ).all()
    assert len(records) == 1
    database.dispose()


def test_run_id_with_different_artifact_path_is_reported(tmp_path):
    database = _database(tmp_path)
    original = {
        "id": "colliding-run",
        "display_name": "Original",
        "name": "source-a/colliding-run",
        "state": "finished",
        "repository_version": {},
    }
    conflicting = {
        **original,
        "display_name": "Conflicting source",
        "name": "source-b/colliding-run",
    }

    assert database.sync_run(original) is True
    assert database.sync_run(conflicting) is False
    assert database.sync_run(conflicting) is False

    with database._session() as session:
        record = session.get(RunIndex, "colliding-run")
    assert record.artifact_name == "source-a/colliding-run"
    conflicts = [
        event for event in database.recent_events(run_id="colliding-run")
        if event["type"] == "run_source_conflict"
    ]
    assert len(conflicts) == 1
    assert conflicts[0]["payload"] == {
        "stored_artifact": "source-a/colliding-run",
        "incoming_artifact": "source-b/colliding-run",
    }
    assert database.status()["source_conflicts"] == 1
    database.dispose()


def test_database_reconnects_after_transient_failure(tmp_path):
    database = _database(tmp_path)
    database._mark_unavailable(RuntimeError("temporary connection reset"))
    database._next_retry_at = 0

    status = database.status()

    assert status["available"] is True
    assert status["error"] is None
    database.dispose()

from dashboard.config import REPO_ROOT
from dashboard.database import DashboardDatabase


def _database(tmp_path):
    path = (tmp_path / "dashboard.db").as_posix()
    database = DashboardDatabase(f"sqlite:///{path}", REPO_ROOT)
    database.initialize(attempts=1, retry_delay_s=0)
    assert database.available is True
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
    }

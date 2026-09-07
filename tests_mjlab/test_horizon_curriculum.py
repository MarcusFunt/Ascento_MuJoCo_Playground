import json
from pathlib import Path
from types import SimpleNamespace

from ascento_mjlab.horizon_curriculum import HORIZON_SCHEDULE_S, HorizonCurriculumRunner


class _Unwrapped:
    def __init__(self, horizon_s: float):
        self.cfg = SimpleNamespace(episode_length_s=horizon_s)

    @property
    def max_episode_length_s(self):
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        return int(self.cfg.episode_length_s * 100)


def _runner(horizon_s=20.0, log_dir=None):
    runner = object.__new__(HorizonCurriculumRunner)
    runner.env = SimpleNamespace(
        unwrapped=_Unwrapped(horizon_s),
        max_episode_length=int(horizon_s * 100),
    )
    runner._completed_in_window = 0
    runner._timeouts_in_window = 0
    runner._successful_windows = 0
    runner._failing_windows = 0
    runner._schedule_index = runner._schedule_index_for(horizon_s)
    runner._stage_windows = 0
    runner._top_horizon_success_windows = 0
    runner._best_top_horizon_timeout_fraction = -1.0
    runner.current_learning_iteration = 123
    runner.logger = SimpleNamespace(log_dir=str(log_dir) if log_dir is not None else None)
    return runner


def test_horizon_schedule_starts_at_the_requested_stage():
    assert HorizonCurriculumRunner._schedule_index_for(20.0) == 0
    assert HorizonCurriculumRunner._schedule_index_for(60.0) == 1
    assert HorizonCurriculumRunner._schedule_index_for(300.0) == 3


def test_horizon_requires_three_qualified_completion_windows(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "_emit_status", lambda **_: None)
    for _ in range(2):
        runner._completed_in_window = 512
        runner._timeouts_in_window = 461
        runner._evaluate_completion_window()
        assert runner.env.unwrapped.cfg.episode_length_s == 20.0

    runner._completed_in_window = 512
    runner._timeouts_in_window = 461
    runner._evaluate_completion_window()

    assert runner.env.unwrapped.cfg.episode_length_s == 60.0


def test_horizon_resets_the_streak_after_a_failing_window(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "_emit_status", lambda **_: None)
    runner._completed_in_window = 512
    runner._timeouts_in_window = 512
    runner._evaluate_completion_window()
    runner._completed_in_window = 512
    runner._timeouts_in_window = 450
    runner._evaluate_completion_window()

    assert runner._successful_windows == 0
    assert runner.env.unwrapped.cfg.episode_length_s == HORIZON_SCHEDULE_S[0]


def test_horizon_demotes_after_sustained_severe_regression(monkeypatch):
    runner = _runner(horizon_s=120.0)
    monkeypatch.setattr(runner, "_emit_status", lambda **_: None)

    for _ in range(runner.required_failure_windows):
        runner._completed_in_window = 512
        runner._timeouts_in_window = 200
        runner._evaluate_completion_window()

    assert runner.env.unwrapped.cfg.episode_length_s == 60.0
    assert runner.env.max_episode_length == 6000
    assert runner._failing_windows == 0


def test_horizon_never_demotes_below_first_stage(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "_emit_status", lambda **_: None)

    for _ in range(3):
        runner._completed_in_window = 512
        runner._timeouts_in_window = 0
        runner._evaluate_completion_window()

    assert runner.env.unwrapped.cfg.episode_length_s == HORIZON_SCHEDULE_S[0]


def test_final_horizon_is_protected_from_training_rollout_demotions(monkeypatch):
    runner = _runner(horizon_s=300.0)
    transitions = []
    monkeypatch.setattr(runner, "_emit_status", lambda **values: transitions.append(values))

    for _ in range(runner.required_failure_windows):
        runner._completed_in_window = 512
        runner._timeouts_in_window = 0
        runner._evaluate_completion_window()

    assert runner.env.unwrapped.cfg.episode_length_s == 300.0
    assert transitions[-1]["transition"] == "protected"
    assert runner._failing_windows == 0


def test_final_horizon_retains_best_training_candidate(monkeypatch, tmp_path):
    runner = _runner(horizon_s=300.0, log_dir=tmp_path)
    monkeypatch.setattr(runner, "_emit_status", lambda **_: None)
    saved = []

    def save(path, infos=None):
        Path(path).write_text("checkpoint", encoding="utf-8")
        saved.append((Path(path), infos))

    monkeypatch.setattr(runner, "save", save)
    for _ in range(runner.required_top_horizon_candidate_windows):
        runner._completed_in_window = 512
        runner._timeouts_in_window = 510
        runner._evaluate_completion_window()

    checkpoint = tmp_path / runner.top_horizon_candidate_name
    metadata = tmp_path / "long_horizon_candidate.json"
    assert checkpoint.read_text(encoding="utf-8") == "checkpoint"
    assert saved[-1][1]["long_horizon_candidate"]["horizon_s"] == 300.0
    assert json.loads(metadata.read_text(encoding="utf-8"))["timeout_fraction"] == 510 / 512

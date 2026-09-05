from ascento_mjlab.evaluation.consistency import check_collection
from ascento_mjlab.evaluation.schema import EpisodeResult, ScenarioSpec


def _scenario():
    return ScenarioSpec(
        scenario_id="suite/family/000000",
        family="family",
        task="Ascento-Balance-Flat",
        horizon_steps=100,
        reset={},
    )


def test_consistent_episode_passes():
    scenario = _scenario()
    result = EpisodeResult(
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        success=True,
        termination_reason="horizon",
        episode_steps=100,
        metrics={
            "episode_time_s": 1.0,
            "path_length": 1.0,
            "net_displacement": 0.5,
            "effort_mean_abs": 2.0,
            "effort_rms": 3.0,
            "effort_max_abs": 4.0,
            "physical_request_max_abs": 40.0,
            "both_supported_fraction": 0.9,
        },
    )
    passed, checks = check_collection([result], [scenario], step_dt=0.01)
    assert passed, [check for check in checks if not check.passed]


def test_telemetry_contradiction_invalidates_collection():
    scenario = _scenario()
    result = EpisodeResult(
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        success=True,
        termination_reason="horizon",
        episode_steps=100,
        metrics={
            "episode_time_s": 1.0,
            "effort_mean_abs": 5.0,
            "effort_rms": 4.0,
            "effort_max_abs": 3.0,
        },
    )
    passed, checks = check_collection([result], [scenario], step_dt=0.01)
    assert not passed
    failed_ids = {check.check_id for check in checks if not check.passed}
    assert "effort_mean_abs_le_rms" in failed_ids
    assert "effort_rms_le_peak" in failed_ids

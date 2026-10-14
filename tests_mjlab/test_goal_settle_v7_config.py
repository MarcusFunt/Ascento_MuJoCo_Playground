"""Isolated V7 task and bounded precision shaping invariants."""

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.task_contract import current_task_contract


def test_v7_alignment_and_isolation():
    v5 = load_env_cfg("Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat", play=False)
    v6 = load_env_cfg("Ascento-Generalist-Locomotion-Goal-Settle-V6-Flat", play=False)
    v7 = load_env_cfg("Ascento-Generalist-Locomotion-Goal-Settle-V7-Flat", play=False)
    assert v5.events["repeated_random_world_targets"].params["target_reached_distance_m"] == 0.10
    assert v6.events["repeated_random_world_targets"].params["target_reached_distance_m"] == 0.08
    assert v7.events["repeated_random_world_targets"].params["target_reached_distance_m"] == 0.09
    assert v7.rewards["target_arrival_settled_stop"].params["target_reached_distance_m"] == 0.09
    params = v7.rewards["target_terminal_precision_quality"].params
    assert params["quality_aggregation"] == "balanced"
    assert params["min_quality_for_credit"] == 0.05
    assert params["max_target_error_m"] == 0.22
    assert set(v7.observations["actor"].terms) == set(v5.observations["actor"].terms)
    assert set(v7.actions) == set(v5.actions)
    assert current_task_contract(v7)["topology_sha256"] != current_task_contract(v6)["topology_sha256"]
    assert "v7" in load_rl_cfg("Ascento-Generalist-Locomotion-Goal-Settle-V7-Flat").experiment_name

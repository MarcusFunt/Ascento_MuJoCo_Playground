"""V6 precision-stop reward/curriculum contract invariants."""

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.task_contract import current_task_contract

OLD = "Ascento-Generalist-Locomotion-Goal-Settle-V5-Flat"
NEW = "Ascento-Generalist-Locomotion-Goal-Settle-V6-Flat"


def test_precision_bonus_and_event_agree_without_mutating_v5():
    old = load_env_cfg(OLD, play=False)
    new = load_env_cfg(NEW, play=False)
    event_old = old.events["repeated_random_world_targets"].params
    event = new.events["repeated_random_world_targets"].params
    assert event_old["target_reached_distance_m"] == 0.10
    assert event["target_reached_distance_m"] == 0.08
    assert new.rewards["target_arrival_settled_stop"].params["target_reached_distance_m"] == 0.08
    assert old.rewards["target_arrival_settled_stop"].params["target_reached_distance_m"] == 0.10
    assert "target_terminal_precision_quality" not in old.rewards
    assert new.rewards["target_terminal_precision_quality"].weight == 14.0
    params = new.rewards["target_terminal_precision_quality"].params
    assert params["hold_s"] == 1.0
    assert params["max_target_error_m"] > event["target_reached_distance_m"]


def test_actor_abi_unchanged_and_reward_abi_distinct():
    old = load_env_cfg(OLD, play=False)
    new = load_env_cfg(NEW, play=False)
    assert set(old.observations["actor"].terms) == set(new.observations["actor"].terms)
    assert set(old.actions) == set(new.actions)
    assert (
        current_task_contract(old)["topology_sha256"]
        != current_task_contract(new)["topology_sha256"]
    )
    assert "v6" in load_rl_cfg(NEW).experiment_name
    assert callable(load_runner_cls(NEW))

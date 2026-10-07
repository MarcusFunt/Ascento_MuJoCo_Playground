import pytest

from ascento_mjlab.mdp import events


@pytest.mark.parametrize(
    ("control_step", "expected"),
    [
        (-10, 0.15),
        (0, 0.15),
        (12_000, 1.075),
        (24_000, 2.0),
        (30_000, 2.0),
    ],
)
def test_linear_curriculum_value_clamps_and_interpolates(control_step, expected):
    assert events.linear_curriculum_value(
        control_step,
        ramp_control_steps=24_000,
        start=0.15,
        end=2.0,
    ) == pytest.approx(expected)


def test_linear_curriculum_value_rejects_nonpositive_ramp():
    with pytest.raises(ValueError, match="ramp_control_steps"):
        events.linear_curriculum_value(
            control_step=0,
            ramp_control_steps=0,
            start=0.15,
            end=2.0,
        )


def test_generalist_locomotion_task_registers_staged_shared_policy():
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import ascento_mjlab.tasks  # noqa: F401
    from ascento_mjlab.task_contract import (
        classify_actor_transfer_compatibility,
        current_task_contract_for_task,
    )

    task_id = "Ascento-Generalist-Locomotion-Flat"
    cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)
    base = load_env_cfg("Ascento-Locomotion-Flat")
    agent = load_rl_cfg(task_id)

    initial = cfg.events["initialize_world_target"].params
    sequence = cfg.events["repeated_random_world_targets"].params
    assert cfg.task_id == task_id
    assert cfg.episode_length_s == pytest.approx(60.0)
    assert initial["min_distance_m"] == pytest.approx(2.0)
    assert initial["max_distance_m"] == pytest.approx(3.0)
    assert initial["curriculum_start_min_target_distance_m"] == pytest.approx(0.15)
    assert initial["curriculum_start_max_target_distance_m"] == pytest.approx(0.35)
    assert initial["curriculum_ramp_control_steps"] == 24_000
    assert sequence["curriculum_start_min_target_distance_m"] == pytest.approx(0.15)
    assert sequence["curriculum_start_max_target_distance_m"] == pytest.approx(0.35)
    assert sequence["min_target_distance_m"] == pytest.approx(2.0)
    assert sequence["max_target_distance_m"] == pytest.approx(3.0)
    assert sequence["curriculum_start_gate_like_fraction"] == pytest.approx(0.10)
    assert sequence["gate_like_fraction"] == pytest.approx(0.25)
    assert sequence["curriculum_ramp_control_steps"] == 24_000
    assert play_cfg.task_id == task_id
    assert "repeated_random_world_targets" in cfg.events
    assert "repeated_random_world_targets" not in play_cfg.events
    assert agent.experiment_name == "ascento_generalist_locomotion_flat"

    source_contract = current_task_contract_for_task("Ascento-Locomotion-Flat")
    target_contract = current_task_contract_for_task(task_id)
    assert source_contract["topology"]["observations"]["actor"] == target_contract["topology"]["observations"]["actor"]
    assert source_contract["topology"]["actions"] == target_contract["topology"]["actions"]
    assert source_contract["topology"] != target_contract["topology"]
    compatibility = classify_actor_transfer_compatibility(source_contract, target_contract)
    assert compatibility.compatible is True

    base_sequence = base.events["repeated_random_world_targets"].params
    assert "curriculum_start_min_target_distance_m" not in base_sequence
    assert base_sequence["min_target_distance_m"] == pytest.approx(2.0)
    assert base_sequence["gate_like_fraction"] == pytest.approx(0.25)

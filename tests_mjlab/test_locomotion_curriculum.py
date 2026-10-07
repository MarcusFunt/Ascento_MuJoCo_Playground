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
    assert initial["stratified_goal_mix"] is True
    assert initial["initial_long_goal_fraction"] == pytest.approx(0.10)
    assert initial["medium_goal_fraction"] == pytest.approx(0.25)
    assert initial["record_training_metrics"] is True
    assert sequence["stratified_goal_mix"] is True
    assert sequence["short_min_distance_m"] == pytest.approx(0.15)
    assert sequence["short_max_distance_m"] == pytest.approx(0.35)
    assert sequence["medium_min_distance_m"] == pytest.approx(0.50)
    assert sequence["medium_max_distance_m"] == pytest.approx(1.50)
    assert sequence["long_min_distance_m"] == pytest.approx(2.0)
    assert sequence["long_max_distance_m"] == pytest.approx(3.0)
    assert sequence["initial_long_goal_fraction"] == pytest.approx(0.10)
    assert sequence["medium_goal_fraction"] == pytest.approx(0.25)
    assert sequence["max_long_goal_fraction"] == pytest.approx(0.40)
    assert sequence["curriculum_start_gate_like_fraction"] == pytest.approx(0.10)
    assert sequence["gate_like_fraction"] == pytest.approx(0.25)
    assert sequence["track_training_metrics"] is True
    assert "generalist_gate_like_arrival_count" in cfg.metrics
    assert "generalist_short_fall_count" in cfg.metrics
    assert "generalist_long_timeout_count" in cfg.metrics
    assert sequence["curriculum_ramp_control_steps"] == 24_000
    assert play_cfg.task_id == task_id
    assert "repeated_random_world_targets" in cfg.events
    assert "repeated_random_world_targets" not in play_cfg.events
    assert agent.experiment_name == "ascento_generalist_locomotion_flat"

    source_contract = current_task_contract_for_task("Ascento-Locomotion-Flat")
    target_contract = current_task_contract_for_task(task_id)
    assert (
        source_contract["topology"]["observations"]["actor"]
        == target_contract["topology"]["observations"]["actor"]
    )
    assert source_contract["topology"]["actions"] == target_contract["topology"]["actions"]
    assert source_contract["topology"] != target_contract["topology"]
    compatibility = classify_actor_transfer_compatibility(source_contract, target_contract)
    assert compatibility.compatible is True

    base_sequence = base.events["repeated_random_world_targets"].params
    assert "curriculum_start_min_target_distance_m" not in base_sequence
    assert base_sequence["min_target_distance_m"] == pytest.approx(2.0)
    assert base_sequence["gate_like_fraction"] == pytest.approx(0.25)


def test_generalist_curriculum_schedule_reports_start_midpoint_and_endpoint():
    schedule = events.generalist_curriculum_values

    start = schedule(0)
    middle = schedule(12_000)
    end = schedule(24_000)

    assert start["target_min_distance_m"] == pytest.approx(0.15)
    assert start["target_max_distance_m"] == pytest.approx(3.0)
    assert start["gate_like_fraction"] == pytest.approx(0.10)
    assert start["progress"] == pytest.approx(0.0)
    assert middle["target_min_distance_m"] == pytest.approx(0.15)
    assert middle["target_max_distance_m"] == pytest.approx(3.0)
    assert middle["gate_like_fraction"] == pytest.approx(0.175)
    assert middle["progress"] == pytest.approx(0.5)
    assert end["target_min_distance_m"] == pytest.approx(0.15)
    assert end["target_max_distance_m"] == pytest.approx(3.0)
    assert end["gate_like_fraction"] == pytest.approx(0.25)
    assert end["progress"] == pytest.approx(1.0)


def test_generalist_gate_share_curriculum_can_be_fixed_at_endpoint(monkeypatch):
    from ascento_mjlab.tasks.generalist_locomotion.env_cfg import (
        ascento_generalist_locomotion_env_cfg,
    )

    monkeypatch.setenv("ASCENTO_GENERALIST_GATE_LIKE_FRACTION", "0.25")
    monkeypatch.setenv("ASCENTO_GENERALIST_GATE_LIKE_START_FRACTION", "0.25")
    cfg = ascento_generalist_locomotion_env_cfg(num_envs=1)

    sequence = cfg.events["repeated_random_world_targets"].params
    metric = cfg.metrics["generalist_curriculum_scheduled_gate_like_fraction"].params
    assert sequence["curriculum_start_gate_like_fraction"] == pytest.approx(0.25)
    assert sequence["gate_like_fraction"] == pytest.approx(0.25)
    assert metric["curriculum_start_gate_like_fraction"] == pytest.approx(0.25)


def test_generalist_gate_recovery_and_phase_metrics_survive_retarget():
    from types import SimpleNamespace

    import torch
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from ascento_mjlab.mdp import metrics

    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_link_vel_w=torch.zeros((1, 3)),
            root_link_lin_vel_b=torch.zeros((1, 3)),
            root_link_ang_vel_b=torch.zeros((1, 3)),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        )
    )
    def write_root_link_velocity_to_sim(velocity, *, env_ids):
        asset.data.root_link_vel_w[env_ids] = velocity

    asset.write_root_link_velocity_to_sim = write_root_link_velocity_to_sim
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.1,
        common_step_counter=0,
        scene={"robot": asset, "left_wheel_contact": contact, "right_wheel_contact": contact},
        ascento_world_target_state={
            "target_xy": torch.tensor([[0.3, 0.0]]),
            "target_yaw": torch.tensor([0.0]),
        },
        termination_manager=SimpleNamespace(
            dones=torch.tensor([False]),
            terminated=torch.tensor([False]),
            time_outs=torch.tensor([False]),
        ),
    )
    sequence = events.RepeatedRandomWorldTargetSequence(None, env)
    state = events.generalist_episode_metrics_state(env)
    sequence._episode_initialized[0] = True
    sequence._gate_like[0] = True
    sequence._episode_elapsed_s[0] = 3.95
    state["gate_like"][0] = True

    def step():
        sequence(
            env,
            torch.tensor([0]),
            gate_push_time_s=4.0,
            gate_retarget_time_s=9.0,
            track_training_metrics=True,
            asset_cfg=SceneEntityCfg("robot"),
        )

    step()
    for _ in range(4):
        step()

    assert state["push_received"].tolist() == [True]
    assert state["recovery_completed"].tolist() == [True]
    sequence._episode_elapsed_s[0] = 8.95

    step()

    assert state["recovery_completed"].tolist() == [True]
    assert state["retargeted"].tolist() == [True]
    assert metrics.generalist_training_metric(
        env, "recovery_count", "gate_like"
    ).tolist() == [1.0]
    assert metrics.generalist_training_metric(
        env, "retarget_count", "gate_like"
    ).tolist() == [1.0]
    assert metrics.generalist_training_metric(
        env, "push_received_count", "gate_like"
    ).tolist() == [1.0]

    env.ascento_world_target_state["target_xy"][0] = asset.data.root_link_pos_w[0, :2]
    for _ in range(4):
        step()

    assert metrics.generalist_training_metric(
        env, "second_target_arrival_count", "gate_like"
    ).tolist() == [1.0]
    assert metrics.generalist_training_metric(
        env, "settled_stop_count", "gate_like"
    ).tolist() == [1.0]
    assert metrics.generalist_training_metric(
        env, "post_arrival_heading_error_samples", "gate_like"
    ).tolist() == [1.0]
    assert metrics.generalist_training_metric(
        env, "post_retarget_settled_heading_error_samples", "gate_like"
    ).tolist() == [1.0]

    events.reset_generalist_episode_metrics(
        env, torch.tensor([0]), initial_target_distance_m=torch.tensor([0.3])
    )
    assert state["recovery_completed"].tolist() == [False]
    assert state["push_received"].tolist() == [False]
    assert state["retargeted"].tolist() == [False]
    assert state["second_target_arrived"].tolist() == [False]
    assert state["settled_stop_count"].tolist() == [0.0]
    assert state["post_retarget_settled_heading_error_samples"].tolist() == [0.0]


def test_generalist_training_metrics_are_slice_conditioned_and_reset_per_episode():
    from types import SimpleNamespace

    import torch

    from ascento_mjlab.mdp import metrics

    env = SimpleNamespace(
        num_envs=6,
        device=torch.device("cpu"),
        common_step_counter=12_000,
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([False, False, False, False, True, False]),
            time_outs=torch.tensor([True, True, True, False, False, True]),
        ),
    )
    state = events.generalist_episode_metrics_state(env)
    events.reset_generalist_episode_metrics(
        env,
        torch.tensor([0, 1, 2, 3, 4, 5]),
        initial_target_distance_m=torch.tensor([0.3, 0.3, 0.7, 0.7, 2.5, 2.5]),
    )
    state["gate_like"][:] = torch.tensor([True, False, True, False, False, True])
    state["arrived"][:] = torch.tensor([True, False, True, True, True, True])
    state["recovery_completed"][:] = torch.tensor([True, False, False, False, False, False])
    state["heading_error_at_arrival_rad"][:] = torch.tensor([0.2, 0.0, 0.25, 0.3, 0.1, 0.2])
    state["arrival_heading_error_sum_rad"][:] = torch.tensor([0.2, 0.0, 0.25, 0.3, 0.1, 0.2])
    state["arrival_heading_error_samples"][:] = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    state["sampled_target_band_counts"][:] = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 2.0],
        ]
    )

    assert metrics.generalist_training_metric(env, "episode_count", "short").tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "episode_count", "medium").tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "episode_count", "long").tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "arrival_count", "short").tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "arrival_count", "medium").tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "arrival_count", "long").tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "recovery_count", "gate_like").tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(
        env, "heading_error_sum_rad", "long"
    ).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    assert metrics.generalist_training_metric(env, "fall_count", "long").tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "timeout_count", "short").tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "sampled_short_goal_targets").tolist() == [
        2.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics.generalist_training_metric(env, "sampled_long_goal_targets").tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        2.0,
    ]

    events.reset_generalist_episode_metrics(
        env, torch.tensor([0]), initial_target_distance_m=torch.tensor([0.7])
    )
    assert state["initial_target_distance_m"].tolist() == pytest.approx(
        [0.7, 0.3, 0.7, 0.7, 2.5, 2.5]
    )
    assert state["arrived"].tolist() == [False, False, True, True, True, True]
    assert state["gate_like"].tolist() == [False, False, True, False, False, True]
    assert state["sampled_target_band_counts"].tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 3.0],
        [0.0, 0.0, 2.0],
    ]


def test_stratified_goal_mix_keeps_every_distance_band_present():
    import torch

    start = events.generalist_goal_mix_fractions(stage=0)
    final = events.generalist_goal_mix_fractions(stage=2)
    assert start == pytest.approx({"short": 0.65, "medium": 0.25, "long": 0.10})
    assert final == pytest.approx({"short": 0.35, "medium": 0.25, "long": 0.40})

    torch.manual_seed(73)
    distances, bands = events.sample_stratified_goal_distances(
        20_000, device=torch.device("cpu"), long_goal_fraction=0.10
    )
    fractions = torch.bincount(bands, minlength=3).float() / bands.numel()
    assert fractions.tolist() == pytest.approx([0.65, 0.25, 0.10], abs=0.015)
    assert torch.all((distances[bands == 0] >= 0.15) & (distances[bands == 0] <= 0.35))
    assert torch.all((distances[bands == 1] >= 0.50) & (distances[bands == 1] <= 1.50))
    assert torch.all((distances[bands == 2] >= 2.00) & (distances[bands == 2] <= 3.00))


def test_long_goal_share_advances_only_after_gate_and_target_success():
    advance = events.advance_generalist_goal_mix_stage
    below_gate = advance(
        0,
        gate_recovery_successes=52,
        gate_episodes=64,
        short_arrival_successes=64,
        short_episodes=64,
        long_arrival_successes=0,
        long_episodes=0,
        minimum_episodes=64,
    )
    assert below_gate == 0

    first_stage = advance(
        0,
        gate_recovery_successes=64,
        gate_episodes=64,
        short_arrival_successes=64,
        short_episodes=64,
        long_arrival_successes=0,
        long_episodes=0,
        minimum_episodes=64,
    )
    assert first_stage == 1

    without_long_evidence = advance(
        1,
        gate_recovery_successes=64,
        gate_episodes=64,
        short_arrival_successes=64,
        short_episodes=64,
        long_arrival_successes=0,
        long_episodes=0,
        minimum_episodes=64,
    )
    assert without_long_evidence == 1

    second_stage = advance(
        1,
        gate_recovery_successes=64,
        gate_episodes=64,
        short_arrival_successes=64,
        short_episodes=64,
        long_arrival_successes=64,
        long_episodes=64,
        minimum_episodes=64,
    )
    assert second_stage == 2


def test_gate_recovery_metric_uses_the_authoritative_stable_envelope():
    from types import SimpleNamespace

    import torch

    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 0.75]]),
            root_link_lin_vel_b=torch.zeros((1, 3)),
            root_link_ang_vel_b=torch.zeros((1, 3)),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.bool)))
    env = SimpleNamespace(
        scene={"robot": asset, "left_wheel_contact": contact, "right_wheel_contact": contact},
        ascento_world_target_state={
            "target_xy": torch.zeros((1, 2)),
            "target_yaw": torch.tensor([0.0]),
        },
    )

    assert events.generalist_gate_recovery_stable(env, asset, torch.tensor([0])).tolist() == [True]
    env.ascento_world_target_state["target_yaw"][0] = 0.20
    assert events.generalist_gate_recovery_stable(env, asset, torch.tensor([0])).tolist() == [False]

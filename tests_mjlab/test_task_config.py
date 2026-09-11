import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import ascento_mjlab.tasks  # noqa: F401
from ascento_mjlab.control_contract import LEG_POSITION_SCALE_RAD, WHEEL_VELOCITY_SCALE_RAD_S
from ascento_mjlab.physics import PHYSICS_PROFILE
from ascento_mjlab.tasks.balance.env_cfg import ascento_balance_env_cfg
from ascento_mjlab.tasks.jump.env_cfg import ascento_jump_env_cfg
from ascento_mjlab.tasks.recovery.env_cfg import ascento_recovery_env_cfg


def test_balance_env_is_six_target_flat_ground():
    cfg = ascento_balance_env_cfg()
    assert cfg.decimation == 5
    assert cfg.sim.mujoco.timestep == 0.002
    assert cfg.scene.terrain is not None
    assert cfg.scene.terrain.terrain_type == "plane"
    env = ManagerBasedRlEnv(cfg, device="cpu")
    obs, _ = env.reset()
    assert obs["actor"].shape[-1] == 40
    assert obs["critic"].shape[-1] == 49
    obs, reward, terminated, truncated, _ = env.step(torch.zeros((cfg.scene.num_envs, 6)))
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()
    assert not terminated.any()
    assert terminated.shape == truncated.shape == (cfg.scene.num_envs,)
    env.close()


def test_balance_action_contract_maps_normalized_targets_and_penalizes_drift():
    cfg = load_env_cfg("Ascento-Balance-Flat")
    cfg.scene.num_envs = 1

    action_cfg = cfg.actions["targets"]
    assert action_cfg.actuator_names == (
        "left_hip",
        "left_knee",
        "left_wheel_joint",
        "right_hip",
        "right_knee",
        "right_wheel_joint",
    )
    assert cfg.rewards["planar_speed"].weight == pytest.approx(-0.05)
    assert cfg.rewards["world_target_proximity"].weight == pytest.approx(4.0)
    assert cfg.rewards["settled_balance"].weight == pytest.approx(1.0)
    assert cfg.rewards["leg_pose_symmetry"].weight == pytest.approx(-2.0)
    assert cfg.rewards["effort"].weight == pytest.approx(-0.8)
    assert cfg.rewards["effort"].params["peak_effort_nm"] == pytest.approx(
        PHYSICS_PROFILE.peak_effort_nm
    )
    assert "world_target_error" in cfg.observations["actor"].terms
    assert "initialize_world_target" in cfg.events
    assert "balance_push" in cfg.events

    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.action_manager.process_action(torch.ones((1, 6)))
    action_term = env.action_manager.get_term("targets")
    assert torch.allclose(
        action_term._position_targets,
        torch.full((1, 4), -3.141592653589793 + LEG_POSITION_SCALE_RAD),
    )
    assert torch.allclose(
        action_term._velocity_targets, torch.full((1, 2), -WHEEL_VELOCITY_SCALE_RAD_S)
    )
    env.close()


def test_balance_rl_config_enforces_normalized_actions_and_instrumented_ppo():
    cfg = load_rl_cfg("Ascento-Balance-Flat")

    assert cfg.clip_actions == 1.0
    assert cfg.algorithm.class_name == "ascento_mjlab.ppo:InstrumentedPPO"
    assert cfg.actor.obs_normalization
    assert cfg.critic.obs_normalization
    assert cfg.algorithm.gamma == pytest.approx(0.998)
    assert cfg.algorithm.lam == pytest.approx(0.97)


def test_velocity_stage_has_no_reward_that_penalizes_its_commands():
    cfg = load_env_cfg("Ascento-Velocity-Flat")

    assert set(cfg.commands) == {"twist", "height"}
    assert "height" not in cfg.rewards
    assert "planar_speed" not in cfg.rewards
    assert "world_target_proximity" not in cfg.rewards
    assert "world_target_error" not in cfg.observations["actor"].terms
    assert "settled_balance" not in cfg.rewards
    assert "leg_pose_symmetry" in cfg.rewards
    assert "balance_push" not in cfg.events
    assert "track_velocity" not in cfg.rewards
    assert "track_linear_velocity" in cfg.rewards
    assert "track_yaw_rate" in cfg.rewards
    assert "track_height" in cfg.rewards
    assert "twist_command" in cfg.observations["actor"].terms
    assert "height_command" in cfg.observations["actor"].terms


def test_recovery_stage_exports_executable_success_metric_and_training_pushes():
    cfg = load_env_cfg("Ascento-Recovery-Flat")
    play_cfg = load_env_cfg("Ascento-Recovery-Flat", play=True)

    assert "recovery_success" in cfg.metrics
    assert "recovery_push" in cfg.events
    assert "balance_push" not in cfg.events
    assert "leg_pose_symmetry" in cfg.rewards
    assert cfg.events["recovery_push"].mode == "interval"
    assert "recovery_push" not in play_cfg.events


@pytest.mark.parametrize(
    "task_id",
    [
        "Ascento-Balance-Flat",
        "Ascento-Velocity-Flat",
        "Ascento-Recovery-Flat",
        "Ascento-Jump-Flat",
    ],
)
def test_all_flat_task_configs_construct_and_step(task_id):
    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    obs, _ = env.reset()
    obs, reward, terminated, truncated, _ = env.step(torch.zeros((1, 6)))
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()
    assert terminated.shape == truncated.shape == (1,)
    env.close()


def test_jump_state_sync_is_first_and_base_rewards_are_phase_aware():
    cfg = load_env_cfg("Ascento-Jump-Flat")

    assert "update_jump_state" not in cfg.events
    reward_names = list(cfg.rewards)
    assert reward_names[0] == "jump_state_sync"
    assert cfg.rewards["height"].func.__name__ == "jump_commanded_height_tracking"
    assert cfg.rewards["angular_rate"].func.__name__ == "jump_angular_rate_penalty"
    assert "planar_speed" not in cfg.rewards
    assert cfg.rewards["lateral_speed"].func.__name__ == "lateral_speed_penalty"
    assert "world_target_proximity" not in cfg.rewards
    assert "settled_balance" not in cfg.rewards
    assert "leg_pose_symmetry" in cfg.rewards
    assert "balance_push" not in cfg.events
    assert cfg.rewards["crouch"].func.__name__ == "jump_crouch"
    assert cfg.rewards["thrust"].func.__name__ == "jump_thrust"
    assert cfg.rewards["track_forward_velocity"].func.__name__ == "track_motion_forward_velocity"
    assert cfg.rewards["track_yaw_rate"].func.__name__ == "track_motion_yaw_rate"
    assert cfg.rewards["recovered_landing"].func.__name__ == "jump_recovered_landing"
    assert cfg.rewards["distance_tracking"].func.__name__ == "jump_distance_tracking"
    assert cfg.rewards["landing_softness"].func.__name__ == "jump_landing_softness"
    assert reward_names.index("jump_state_sync") < reward_names.index("takeoff")
    assert reward_names.index("jump_state_sync") < reward_names.index("landing")
    assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.01)


def test_dense_specialist_shaping_can_be_disabled_for_ablation(monkeypatch):
    monkeypatch.setenv("ASCENTO_DISABLE_DENSE_SHAPING", "1")
    recovery = ascento_recovery_env_cfg()
    jump = ascento_jump_env_cfg()
    assert "recovery_dwell" not in recovery.rewards
    assert "post_landing_stability" not in jump.rewards


def test_balance_experiment_overrides_change_reward_weights(monkeypatch):
    monkeypatch.setenv("ASCENTO_BALANCE_DRIFT_PENALTY_SCALE", "2.5")
    monkeypatch.setenv("ASCENTO_BALANCE_STABILIZATION_WEIGHT", "1.75")
    cfg = ascento_balance_env_cfg()
    assert cfg.rewards["planar_speed"].weight == pytest.approx(-0.125)
    assert cfg.rewards["settled_balance"].weight == pytest.approx(1.75)


def test_jump_observation_contains_phase_and_remaining_distance():
    cfg = load_env_cfg("Ascento-Jump-Flat", play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        obs, _ = env.reset()
        # Balance actor (38 after removing its target term) + motion command (6) + jump state (6).
        assert obs["actor"].shape[-1] == 50
    finally:
        env.close()


def test_custom_sim_timestep_reaches_actuator():
    cfg = load_env_cfg("Ascento-Balance-Flat")
    cfg.scene.num_envs = 1
    cfg.sim.mujoco.timestep = 0.007

    env = ManagerBasedRlEnv(cfg, device="cpu")

    assert {act._physics_dt for act in env.scene["robot"].actuators} == {0.007}
    env.close()

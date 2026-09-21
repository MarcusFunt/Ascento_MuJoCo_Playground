from types import SimpleNamespace

import torch

from ascento_mjlab.evaluation.runner import (
    _apply_world_target_offset,
    _evaluation_env_cfg,
    _refresh_exact_observation_history,
    _reset_finished_slots,
    _stationary_quality_mask,
    physics_timestep,
)
from ascento_mjlab.mdp.events import world_target_xy, world_target_yaw


class _ObservationManager:
    def __init__(self):
        self.reset_ids = None
        self.compute_ids = None

    def reset(self, env_ids):
        self.reset_ids = env_ids.clone()

    def compute(self, *, update_history, env_ids):
        assert update_history is True
        self.compute_ids = env_ids.clone()
        return {"actor": torch.arange(len(env_ids), dtype=torch.float32)[:, None]}


def test_exact_reset_reseeds_observation_history_from_resolved_state():
    manager = _ObservationManager()
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        observation_manager=manager,
        obs_buf=None,
    )

    _refresh_exact_observation_history(env)

    expected = torch.tensor([0, 1, 2])
    assert torch.equal(manager.reset_ids, expected)
    assert torch.equal(manager.compute_ids, expected)
    assert torch.equal(env.obs_buf["actor"].flatten(), torch.tensor([0.0, 1.0, 2.0]))


def test_finished_vector_slots_are_reset_before_next_step():
    reset_calls = []
    policy_calls = []
    env = SimpleNamespace(reset=lambda *, env_ids: reset_calls.append(env_ids.clone()))
    policy = SimpleNamespace(reset=lambda env_ids: policy_calls.append(env_ids.clone()))
    finish = torch.tensor([False, True, False, True])

    _reset_finished_slots(env, policy, finish)

    expected = torch.tensor([1, 3])
    assert len(reset_calls) == len(policy_calls) == 1
    assert torch.equal(reset_calls[0], expected)
    assert torch.equal(policy_calls[0], expected)


def test_no_finished_slots_do_not_trigger_reset():
    env = SimpleNamespace(reset=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)))
    policy = SimpleNamespace(reset=lambda *args: (_ for _ in ()).throw(AssertionError(args)))

    _reset_finished_slots(env, policy, torch.zeros(4, dtype=torch.bool))


def test_physics_timestep_uses_mjlab_mujoco_config():
    cfg = SimpleNamespace(sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.002)))

    assert physics_timestep(cfg) == 0.002


def test_world_target_offset_is_rotated_by_current_yaw():
    """A relative target remains valid under the evaluator's yaw randomization."""
    yaw = torch.tensor(torch.pi / 2.0)
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_pos_w=torch.tensor([[1.0, 2.0, 0.75]]),
                    root_link_quat_w=torch.tensor(
                        [[torch.cos(yaw / 2.0), 0.0, 0.0, torch.sin(yaw / 2.0)]]
                    ),
                )
            )
        }
    )

    _apply_world_target_offset(env, [(0, (0.20, 0.0, 0.0))])

    assert torch.allclose(world_target_xy(env)[0], torch.tensor([1.0, 2.20]))
    assert torch.allclose(world_target_yaw(env), torch.tensor([yaw]))


def test_evaluation_config_omits_locomotion_training_sequence():
    cfg = _evaluation_env_cfg("Ascento-Locomotion-Flat", capacity=2, max_horizon=100)

    assert "settle_triggered_sequence" not in cfg.events
    assert cfg.scene.num_envs == 2
    assert cfg.auto_reset is False


def test_stationary_quality_only_samples_currently_quiet_slots():
    active = torch.tensor([True, True, False])
    stable_now = torch.tensor([True, False, True])

    assert torch.equal(
        _stationary_quality_mask(active, stable_now),
        torch.tensor([True, False, False]),
    )

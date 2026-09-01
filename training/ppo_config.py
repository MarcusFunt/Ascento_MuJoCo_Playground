"""Shared stage/environment/PPO configuration for the Ascento curriculum."""
from __future__ import annotations

from dataclasses import asdict

from brax.training.agents.ppo import networks as ppo_networks

from ascento import AscentoBalance, AscentoJump, AscentoRecovery
from ascento.curriculum import StageSpec, stage_by_name
from training.bounded_distribution import BoundedNormalTanhDistribution


def build_environment(stage_name: str, episode_length: int = 600):
    """Builds one direct-torque environment without changing its IO schema."""
    stage = stage_by_name(stage_name)
    kwargs = dict(
        episode_length=episode_length,
        max_vx=stage.max_vx,
        max_yaw_rate=stage.max_yaw_rate,
        height_range=stage.height_range,
        reset_tilt=stage.disturbance_tilt,
        jump_probability=stage.jump_probability,
        max_jump_height=stage.max_jump_height,
        max_jump_distance=stage.max_jump_distance,
    )
    env_type = {
        "balance": AscentoBalance,
        "recovery": AscentoRecovery,
        "jump": AscentoJump,
    }[stage.env]
    return env_type(**kwargs), stage


def network_factory(hidden_sizes=(512, 256, 128)):
    def factory(obs_size, action_size, preprocess_observations_fn):
        ppo_network = ppo_networks.make_ppo_networks(
            obs_size,
            action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=hidden_sizes,
            value_hidden_layer_sizes=hidden_sizes,
            policy_obs_key="state",
            value_obs_key="state",
        )
        return ppo_network.replace(
            parametric_action_distribution=BoundedNormalTanhDistribution(action_size)
        )
    return factory


def default_ppo_kwargs(smoke: bool = False) -> dict:
    """Brax PPO baseline; task stages vary environments, not controller type."""
    return dict(
        # Direct torque exploration is considerably less forgiving than the
        # position-controlled Brax examples.  A single conservative update per
        # rollout avoids driving the policy/value heads non-finite during the
        # initial, high-variance balance rollouts.
        learning_rate=3e-5,
        entropy_cost=1e-3,
        discounting=0.99,
        unroll_length=20,
        batch_size=256 if not smoke else 64,
        num_minibatches=4,
        num_updates_per_batch=1,
        # Actor features are clipped to physical ranges, so the streaming
        # normalizer can now safely remove scale differences across channels.
        normalize_observations=True,
        reward_scaling=0.01,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=0.5,
    )


def stage_manifest(stage: StageSpec) -> dict:
    return asdict(stage)

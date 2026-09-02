"""Shared stage/environment/PPO configuration for the Ascento curriculum."""
from __future__ import annotations

from dataclasses import asdict, replace
import math

import jax.numpy as jp

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
        reset_angular_velocity=stage.reset_angular_velocity,
        reset_linear_velocity=stage.reset_linear_velocity,
        reset_leg_variation=stage.reset_leg_variation,
        reset_wheel_velocity=stage.reset_wheel_velocity,
        action_scale=stage.action_scale,
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


def network_factory(hidden_sizes=(512, 256, 128), initial_noise_std: float = 0.10):
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
        # ``tanh_normal`` normally initializes both action means and scales
        # from a generic MLP.  For a free-standing robot that means the first
        # rollout injects arbitrary torques even though zero torque is the
        # known static equilibrium.  Initialize only the final action head to
        # zero mean and the requested bounded-distribution standard deviation.
        base_policy = ppo_network.policy_network
        requested_std = min(max(float(initial_noise_std), 0.0201), 0.1999)
        scale_logit = math.log((requested_std - 0.02) / (0.20 - requested_std))
        final_key = f"hidden_{len(hidden_sizes)}"

        def init_policy(key):
            params = base_policy.init(key)
            final = params["params"][final_key]
            bias = jp.zeros_like(final["bias"])
            bias = bias.at[action_size:].set(scale_logit)
            final = dict(final, kernel=jp.zeros_like(final["kernel"]), bias=bias)
            return dict(params, params=dict(params["params"], **{final_key: final}))

        ppo_network = ppo_network.replace(
            policy_network=replace(ppo_network.policy_network, init=init_policy)
        )
        return ppo_network.replace(
            parametric_action_distribution=BoundedNormalTanhDistribution(action_size)
        )
    return factory


def default_ppo_kwargs(stage, smoke: bool = False) -> dict:
    """Brax PPO baseline; task stages vary environments, not controller type."""
    return dict(
        # The initial policy and exploration are bounded at the source, so PPO
        # can now make enough optimizer progress to change the torque head.
        learning_rate=stage.learning_rate,
        # Exploration is deliberately stage-specific.  A standing robot needs
        # its torque noise to contract before it can demonstrate a stable
        # rollout; larger disturbances arrive only in later stages.
        entropy_cost=stage.entropy_cost,
        discounting=0.99,
        unroll_length=20,
        batch_size=256 if not smoke else 64,
        num_minibatches=4,
        num_updates_per_batch=stage.updates_per_batch,
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

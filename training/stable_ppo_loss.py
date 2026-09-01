"""A guarded PPO loss for high-variance direct-torque MJX rollouts."""
from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jp

from brax.training import types
from brax.training.agents.ppo import losses as brax_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params


def compute_ppo_loss(
    params: Params,
    normalizer_params: Any,
    data: types.Transition,
    rng: jp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.99,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.2,
    normalize_advantage: bool = True,
) -> Tuple[jp.ndarray, types.Metrics]:
    """PPO loss with finite ratio, advantage, and critic targets.

    The clipping mirrors the practical safeguards in wheel-legged PPO stacks:
    it bounds the likelihood ratio and critic residual before an optimizer step
    can corrupt a policy from an unusually energetic early rollout.
    """
    data = jax.tree_util.tree_map(lambda value: jp.swapaxes(value, 0, 1), data)
    distribution = ppo_network.parametric_action_distribution
    policy_logits = ppo_network.policy_network.apply(normalizer_params, params.policy, data.observation)
    baseline = ppo_network.value_network.apply(normalizer_params, params.value, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda value: value[-1], data.next_observation)
    bootstrap_value = ppo_network.value_network.apply(normalizer_params, params.value, terminal_obs)
    rewards = jp.nan_to_num(data.reward * reward_scaling, nan=0.0, posinf=10.0, neginf=-10.0)
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1.0 - data.discount) * (1.0 - truncation)
    vs, advantages = brax_losses.compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )
    if normalize_advantage:
        advantages = (advantages - jp.mean(advantages)) / (jp.std(advantages) + 1e-6)
    advantages = jp.clip(jp.nan_to_num(advantages), -5.0, 5.0)
    target_log_prob = distribution.log_prob(policy_logits, data.extras["policy_extras"]["raw_action"])
    behavior_log_prob = data.extras["policy_extras"]["log_prob"]
    ratio = jp.exp(jp.clip(target_log_prob - behavior_log_prob, -10.0, 10.0))
    policy_loss = -jp.mean(jp.minimum(ratio * advantages, jp.clip(ratio, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages))
    value_error = jp.clip(jp.nan_to_num(vs - baseline), -10.0, 10.0)
    value_loss = 0.25 * jp.mean(jp.square(value_error))
    # The bounded rollout distribution normally keeps entropy in this range;
    # retain an explicit final bound so a future distribution change cannot
    # convert a finite saturated logit into an optimizer-dominating loss.
    entropy = jp.mean(jp.clip(jp.nan_to_num(distribution.entropy(policy_logits, rng)), -20.0, 20.0))
    entropy_loss = -entropy_cost * entropy
    total_loss = policy_loss + value_loss + entropy_loss
    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": value_loss,
        "entropy_loss": entropy_loss,
    }

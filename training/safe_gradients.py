"""Numerically guarded PPO gradient updates for MJX direct-torque training."""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jp
import optax

from brax.training import gradients


def _all_finite(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jp.all(jp.stack([jp.all(jp.isfinite(leaf)) for leaf in leaves]))


def _global_norm(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jp.sqrt(sum(jp.sum(jp.square(leaf)) for leaf in leaves))


def _finite_tree(tree):
    return jax.tree_util.tree_map(
        lambda leaf: jp.nan_to_num(leaf, nan=0.0, posinf=1e4, neginf=-1e4), tree
    )


def _select_tree(use_new: jax.Array, old, new):
    return jax.tree_util.tree_map(
        lambda old_leaf, new_leaf: jp.where(use_new, new_leaf, old_leaf), old, new
    )


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Brax-compatible update wrapper that makes invalid updates no-ops.

    MJX direct-torque rollouts occasionally create a non-finite intermediate
    gradient after a sharp early policy change.  The stock Brax wrapper applies
    that gradient directly, irreversibly corrupting both parameters and Adam's
    moments.  This wrapper keeps the last finite state and records the event.
    """
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def update(*args, optimizer_state):
        value, grads = loss_and_pgrad_fn(*args)
        loss, metrics = value if has_aux else (value, {})
        grad_norm = _global_norm(grads)
        # A finite but enormous update is just as destructive as a NaN for a
        # direct-torque rollout.  Optax clips ordinary gradients later; this
        # bound rejects only pathological samples before Adam moments change.
        valid = jp.isfinite(loss) & (jp.abs(loss) <= 1e3) & _all_finite(grads) & (grad_norm <= 1e3)
        safe_grads = _finite_tree(grads)
        updates, candidate_optimizer_state = optimizer.update(
            safe_grads, optimizer_state, args[0]
        )
        candidate_params = optax.apply_updates(args[0], updates)
        valid = valid & _all_finite(candidate_params) & _all_finite(candidate_optimizer_state)
        params = _select_tree(valid, args[0], candidate_params)
        next_optimizer_state = _select_tree(valid, optimizer_state, candidate_optimizer_state)

        if has_aux:
            safe_metrics = _finite_tree(metrics)
            safe_metrics = dict(
                safe_metrics,
                invalid_update=1.0 - valid.astype(jp.float32),
                gradient_norm=jp.nan_to_num(grad_norm, nan=1e6, posinf=1e6, neginf=1e6),
            )
            safe_value = (jp.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4), safe_metrics)
        else:
            safe_value = jp.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        return safe_value, params, next_optimizer_state

    return update

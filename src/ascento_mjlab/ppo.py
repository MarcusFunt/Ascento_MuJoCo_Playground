"""Ascento PPO instrumentation built on the pinned RSL-RL implementation.

RSL-RL uses KL internally for its adaptive learning-rate schedule but does not
emit KL or clip fraction in its default loss dictionary.  This subclass keeps
the upstream optimizer/update implementation intact, then evaluates the final
updated policy against the rollout policy to expose stable full-rollout
post-update diagnostics to TensorBoard.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class InstrumentedPPO(PPO):
  """PPO with post-update KL and clip-fraction telemetry."""

  def update(self) -> dict[str, float]:
    loss_dict = super().update()

    # RolloutStorage.clear() only resets the write cursor; the rollout tensors
    # remain valid until the next collection phase.  Evaluate the newly updated
    # actor against those old-policy samples without changing optimizer state.
    with torch.inference_mode():
      observations = self.storage.observations.flatten(0, 1)
      actions = self.storage.actions.flatten(0, 1)
      old_log_prob = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1)
      old_distribution_params = tuple(
        parameter.flatten(0, 1) for parameter in self.storage.distribution_params
      )

      self.actor(observations, stochastic_output=True)
      new_log_prob = self.actor.get_output_log_prob(actions)
      new_distribution_params = self.actor.output_distribution_params

      ratio = torch.exp(new_log_prob - old_log_prob)
      clip_fraction = torch.mean(
        ((ratio < (1.0 - self.clip_param)) | (ratio > (1.0 + self.clip_param))).float()
      )
      kl = torch.mean(
        self.actor.get_kl_divergence(old_distribution_params, new_distribution_params)
      )

      if self.is_multi_gpu:
        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(clip_fraction, op=torch.distributed.ReduceOp.SUM)
        kl /= self.gpu_world_size
        clip_fraction /= self.gpu_world_size

    loss_dict["kl"] = float(kl.item())
    loss_dict["clip_fraction"] = float(clip_fraction.item())
    return loss_dict

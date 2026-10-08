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
from tensordict import TensorDict

from .control_contract import LEG_POSITION_SCALE_RAD, WHEEL_VELOCITY_SCALE_RAD_S


def apply_actor_reference_mse(
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    coefficient: float,
    max_gradient_norm: float | None = None,
) -> dict[str, float]:
    """Apply one deterministic-mean MSE step to actor parameters only.

    A zero coefficient returns before touching the actor, optimizer, or their
    gradients. RSL-RL's actor and critic share one optimizer, so the reference
    step clears gradients to ``None`` and backpropagates through only the actor;
    optimizer state for critic parameters is consequently left untouched.
    """
    coefficient = float(coefficient)
    if not torch.isfinite(torch.tensor(coefficient)) or coefficient < 0.0:
        raise ValueError("reference coefficient must be finite and nonnegative")
    zero_metrics = {
        "reference_aux_loss": 0.0,
        "reference_coefficient": coefficient,
        "reference_sample_count": 0.0,
        "anchor_student_teacher_action_rms": 0.0,
        "reference_actor_gradient_norm": 0.0,
        "reference_policy_std_delta_rms": 0.0,
    }
    if coefficient == 0.0:
        return zero_metrics
    if observations.ndim != 2 or teacher_actions.ndim != 2:
        raise ValueError("reference observations and actions must be rank-2 batches")
    if observations.shape[0] < 1 or teacher_actions.shape[0] != observations.shape[0]:
        raise ValueError("reference observations and actions must have aligned nonempty batches")
    if not torch.isfinite(observations).all() or not torch.isfinite(teacher_actions).all():
        raise ValueError("reference observations and actions must be finite")

    actor_input: object = observations
    actor_groups = getattr(actor, "obs_groups", None)
    if actor_groups is not None:
        if len(actor_groups) != 1:
            raise ValueError(
                "anchor replay currently requires one concatenated actor observation group"
            )
        actor_input = TensorDict(
            {actor_groups[0]: observations}, batch_size=[observations.shape[0]]
        )

    if actor_groups is None:
        student_actions = actor(actor_input)
    else:
        student_actions = actor(actor_input, stochastic_output=False)  # type: ignore[call-arg]
    if student_actions.shape != teacher_actions.shape:
        raise ValueError(
            f"student action shape {tuple(student_actions.shape)} does not match teacher "
            f"shape {tuple(teacher_actions.shape)}"
        )
    teacher_actions = teacher_actions.to(device=student_actions.device, dtype=student_actions.dtype)
    mse = torch.mean((student_actions - teacher_actions.detach()).square())
    auxiliary_loss = coefficient * mse
    std_parameters = [
        parameter
        for name, parameter in actor.named_parameters()
        if name.endswith("std_param") or name.endswith("log_std_param")
    ]
    std_before = [parameter.detach().clone() for parameter in std_parameters]

    optimizer.zero_grad(set_to_none=True)
    auxiliary_loss.backward()
    actor_parameters = [parameter for parameter in actor.parameters() if parameter.grad is not None]
    if not actor_parameters:
        raise RuntimeError("reference loss produced no actor gradients")
    gradient_norm = torch.linalg.vector_norm(
        torch.stack(
            [torch.linalg.vector_norm(parameter.grad.detach()) for parameter in actor_parameters]
        )
    )
    if max_gradient_norm is not None:
        if not torch.isfinite(torch.tensor(max_gradient_norm)) or max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be finite and positive")
        torch.nn.utils.clip_grad_norm_(actor_parameters, float(max_gradient_norm))
    optimizer.step()
    if std_parameters:
        std_delta_rms = torch.sqrt(
            torch.mean(
                torch.cat(
                    [
                        (parameter.detach() - before).reshape(-1).square()
                        for parameter, before in zip(std_parameters, std_before, strict=True)
                    ]
                )
            )
        )
    else:
        std_delta_rms = mse.detach().new_zeros(())

    return {
        "reference_aux_loss": float(auxiliary_loss.detach().item()),
        "reference_coefficient": coefficient,
        "reference_sample_count": float(observations.shape[0]),
        "anchor_student_teacher_action_rms": float(torch.sqrt(mse.detach()).item()),
        "reference_actor_gradient_norm": float(gradient_norm.item()),
        "reference_policy_std_delta_rms": float(std_delta_rms.item()),
    }


class InstrumentedPPO(PPO):
    """PPO with post-update KL and clip-fraction telemetry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_replay = None
        self.reference_coefficient = 0.0
        self.reference_batch_size = 0

    def configure_anchor_replay(self, dataset, *, coefficient: float, batch_size: int) -> None:
        """Attach a prevalidated immutable replay dataset to the training actor."""
        if coefficient <= 0.0:
            raise ValueError("anchor replay configuration requires a positive coefficient")
        if batch_size < 2:
            raise ValueError("anchor replay batch size must be at least 2")
        self.anchor_replay = dataset
        self.reference_coefficient = float(coefficient)
        self.reference_batch_size = int(batch_size)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()

        # Keep explicit aliases for experiment telemetry while retaining the
        # upstream names consumed by existing dashboards and result tooling.
        loss_dict["ppo_actor_loss"] = float(loss_dict.get("surrogate", 0.0))
        loss_dict["value_loss"] = float(loss_dict.get("value", 0.0))

        if self.reference_coefficient > 0.0:
            if self.anchor_replay is None:
                raise RuntimeError("positive anchor replay coefficient has no validated dataset")
            observations, teacher_actions = self.anchor_replay.sample(
                self.reference_batch_size,
                device=self.device,
            )
            loss_dict.update(
                apply_actor_reference_mse(
                    self.actor,
                    self.optimizer,
                    observations,
                    teacher_actions,
                    coefficient=self.reference_coefficient,
                    max_gradient_norm=self.max_grad_norm,
                )
            )
        else:
            loss_dict.update(
                {
                    "reference_aux_loss": 0.0,
                    "reference_coefficient": 0.0,
                    "reference_sample_count": 0.0,
                    "anchor_student_teacher_action_rms": 0.0,
                    "reference_actor_gradient_norm": 0.0,
                    "reference_policy_std_delta_rms": 0.0,
                }
            )

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

        # Rollout actions are normalized structured targets, never torque commands.
        legs = actions[:, (0, 1, 3, 4)].clamp(-1.0, 1.0) * LEG_POSITION_SCALE_RAD
        wheels = actions[:, (2, 5)].clamp(-1.0, 1.0) * WHEEL_VELOCITY_SCALE_RAD_S
        loss_dict["leg_target_offset_rms_rad"] = float(torch.sqrt(torch.mean(legs.square())).item())
        loss_dict["wheel_target_velocity_rms_rad_s"] = float(
            torch.sqrt(torch.mean(wheels.square())).item()
        )
        loss_dict["command_saturation_fraction"] = float(
            torch.mean((actions.abs() >= 1.0 - 1.0e-4).float()).item()
        )
        return loss_dict

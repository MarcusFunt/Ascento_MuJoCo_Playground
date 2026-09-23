"""Viewer-only capture of policy observations, activations, value, and scale."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch
from torch import nn

from ..control_contract import LEG_POSITION_SCALE_RAD
from ..robot_cfg import JOINT_NAMES
from .introspection_contract import PolicyRuntimeHandles
from .replay import TransitionSnapshot


@dataclass(frozen=True)
class PhysicalTarget:
    channel: str
    kind: str
    value: float
    unit: str
    joint_limit_clipped: bool


@dataclass(frozen=True)
class ActionPipelineSnapshot:
    actor_output: torch.Tensor
    wrapper_clipped: torch.Tensor
    processed_action: torch.Tensor
    wrapper_clipped_flags: tuple[bool, ...]
    action_term_clipped_flags: tuple[bool, ...]
    targets: tuple[PhysicalTarget, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_output": self.actor_output.tolist(),
            "wrapper_clipped": self.wrapper_clipped.tolist(),
            "processed_action": self.processed_action.tolist(),
            "wrapper_clipped_flags": list(self.wrapper_clipped_flags),
            "action_term_clipped_flags": list(self.action_term_clipped_flags),
            "targets": [
                {
                    "channel": item.channel,
                    "kind": item.kind,
                    "value": item.value,
                    "unit": item.unit,
                    "joint_limit_clipped": item.joint_limit_clipped,
                }
                for item in self.targets
            ],
        }


@dataclass(frozen=True)
class StructuredActionStepSnapshot:
    processed_action: torch.Tensor
    position_targets: torch.Tensor
    velocity_targets: torch.Tensor
    leg_slots: tuple[int, ...]
    wheel_slots: tuple[int, ...]
    channel_names: tuple[str, ...]
    joint_limit_clipped: tuple[bool, ...]


class StructuredActionObserver:
    """Viewer-only tap that snapshots actual action outputs before reset clears them."""

    def __init__(self, action_term: Any) -> None:
        self.action_term = action_term
        self.latest: StructuredActionStepSnapshot | None = None
        self._original_process_actions = action_term.process_actions

        def process_actions(actions: torch.Tensor) -> None:
            self._original_process_actions(actions)
            self.latest = self._snapshot()

        self._process_actions_wrapper = process_actions
        action_term.process_actions = process_actions

    def close(self) -> None:
        if getattr(self.action_term, "process_actions", None) is self._process_actions_wrapper:
            self.action_term.process_actions = self._original_process_actions

    def _snapshot(self) -> StructuredActionStepSnapshot:
        term = self.action_term
        processed = _first_environment_vector(term.raw_action)
        position_targets = _first_environment_vector(term.position_targets)
        velocity_targets = _first_environment_vector(term.velocity_targets)
        leg_slots = _as_indices(term._leg_slots)
        wheel_slots = _as_indices(term._wheel_slots)
        names = tuple(str(name) for name in getattr(term.cfg, "actuator_names", ()))
        if len(names) != processed.numel():
            names = tuple(f"action_{index}" for index in range(processed.numel()))
        joint_limit_clipped = _joint_limit_clipping(
            term,
            processed,
            position_targets,
            leg_slots,
        )
        return StructuredActionStepSnapshot(
            processed_action=processed,
            position_targets=position_targets,
            velocity_targets=velocity_targets,
            leg_slots=leg_slots,
            wheel_slots=wheel_slots,
            channel_names=names,
            joint_limit_clipped=joint_limit_clipped,
        )


@dataclass(frozen=True)
class PolicyIntrospectionFrame:
    sequence_id: int
    policy_generation: int
    checkpoint: str
    captured_at: float
    raw_observation: torch.Tensor
    normalized_observation: torch.Tensor
    actor_output: torch.Tensor
    hidden_activations: dict[str, torch.Tensor]
    critic_value: float | None
    policy_std: torch.Tensor | None
    policy_std_parameter_count: int | None = None
    policy_std_state_dependent: bool | None = None
    jacobian: torch.Tensor | None = None
    jacobian_calculated_at_sequence: int | None = None
    jacobian_age_steps: int | None = None
    jacobian_age_ms: float | None = None
    action_pipeline: ActionPipelineSnapshot | None = None
    transition: TransitionSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "policy_generation": self.policy_generation,
            "checkpoint": self.checkpoint,
            "captured_at": self.captured_at,
            "raw_observation": self.raw_observation.tolist(),
            "normalized_observation": self.normalized_observation.tolist(),
            "actor_output": self.actor_output.tolist(),
            "hidden_activations": {
                name: value.tolist() for name, value in self.hidden_activations.items()
            },
            "critic_value": self.critic_value,
            "policy_std": self.policy_std.tolist() if self.policy_std is not None else None,
            "policy_std_parameter_count": self.policy_std_parameter_count,
            "policy_std_state_dependent": self.policy_std_state_dependent,
            "jacobian": self.jacobian.tolist() if self.jacobian is not None else None,
            "jacobian_calculated_at_sequence": self.jacobian_calculated_at_sequence,
            "jacobian_age_steps": self.jacobian_age_steps,
            "jacobian_age_ms": self.jacobian_age_ms,
            "action_pipeline": (
                self.action_pipeline.to_dict() if self.action_pipeline is not None else None
            ),
            "transition": self.transition.to_dict() if self.transition is not None else None,
        }


@dataclass(frozen=True)
class CapturedPolicyAction:
    """Original device tensor returned to the simulator plus its CPU snapshot."""

    action: torch.Tensor
    frame: PolicyIntrospectionFrame


class PolicyIntrospector:
    """Observe a single authoritative actor forward without changing its result."""

    def __init__(
        self,
        handles: PolicyRuntimeHandles,
        *,
        checkpoint: str = "",
        jacobian_hz: float = 2,
        policy_generation: int = 1,
    ) -> None:
        self.handles = handles
        self.checkpoint = str(checkpoint)
        self.policy_generation = int(policy_generation)
        self._hook_handles: list[Any] = []
        self._raw: torch.Tensor | None = None
        self._normalized: torch.Tensor | None = None
        self._actor_logits: torch.Tensor | None = None
        self._activations: dict[str, torch.Tensor] = {}
        self._activation_counts: dict[str, int] = {}
        self._capturing = False
        self.jacobian_cadence = JacobianCadence(jacobian_hz)
        self._last_jacobian: torch.Tensor | None = None
        self._last_jacobian_sequence: int | None = None
        self._last_jacobian_time: float | None = None
        self._install_hooks()

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def capture_action(
        self,
        observations: Any,
        policy_forward: Callable[[], torch.Tensor],
        *,
        sequence_id: int,
    ) -> CapturedPolicyAction:
        """Run and return the policy's sole authoritative output plus a snapshot."""

        self._raw = None
        self._normalized = None
        self._actor_logits = None
        self._activations = {}
        self._activation_counts = {}

        self._capturing = True
        try:
            actor_output = policy_forward()
        finally:
            self._capturing = False
        if not isinstance(actor_output, torch.Tensor):
            raise TypeError("viewer actor output must be a tensor")
        raw = (
            self._raw
            if self._raw is not None
            else _select_observation(observations, self.handles.actor_obs_groups)
        )
        normalized = self._normalized if self._normalized is not None else raw
        logits = self._actor_logits

        critic_value: float | None = None
        critic = self.handles.critic
        if critic is not None:
            self._capturing = True
            try:
                with torch.inference_mode():
                    critic_output = critic(observations)
            finally:
                self._capturing = False
            if not isinstance(critic_output, torch.Tensor) or critic_output.numel() != 1:
                raise ValueError("viewer critic must produce one scalar value for one environment")
            critic_value = float(critic_output.detach().reshape(-1)[0].cpu().item())

        policy_std = self._policy_std(logits)
        frame = PolicyIntrospectionFrame(
            sequence_id=int(sequence_id),
            policy_generation=self.policy_generation,
            checkpoint=self.checkpoint,
            captured_at=time.time(),
            raw_observation=_first_environment_vector(raw),
            normalized_observation=_first_environment_vector(normalized),
            actor_output=_first_environment_vector(actor_output),
            hidden_activations={
                name: _first_environment_vector(value)
                for name, value in self._activations.items()
            },
            critic_value=critic_value,
            policy_std=_first_environment_vector(policy_std) if policy_std is not None else None,
            policy_std_parameter_count=self._std_parameter_count(),
            policy_std_state_dependent=self._std_state_dependent(),
        )
        return CapturedPolicyAction(action=actor_output, frame=frame)

    def maybe_calculate_jacobian(
        self,
        frame: PolicyIntrospectionFrame,
        *,
        now: float | None = None,
    ) -> PolicyIntrospectionFrame:
        """Calculate at configured cadence and annotate the last value's age."""

        current = time.monotonic() if now is None else float(now)
        if self.jacobian_cadence.should_calculate(now=current):
            self._last_jacobian = compute_policy_jacobian(
                self.handles,
                frame.raw_observation,
            ).to(device="cpu").clone()
            self._last_jacobian_sequence = frame.sequence_id
            self._last_jacobian_time = current
        if self._last_jacobian is None or self._last_jacobian_sequence is None:
            return frame
        age_seconds = max(0.0, current - (self._last_jacobian_time or current))
        return replace(
            frame,
            jacobian=self._last_jacobian,
            jacobian_calculated_at_sequence=self._last_jacobian_sequence,
            jacobian_age_steps=max(0, frame.sequence_id - self._last_jacobian_sequence),
            jacobian_age_ms=age_seconds * 1000.0,
        )

    def _install_hooks(self) -> None:
        normalizer = self.handles.actor_normalizer
        if normalizer is not None:
            self._hook_handles.append(
                normalizer.register_forward_hook(self._capture_normalizer)
            )
        self._hook_handles.append(self.handles.actor_body.register_forward_hook(self._capture_logits))
        self._install_activation_hooks(self.handles.actor_body, "actor", self.handles.actor_body_name)
        if self.handles.critic_body is not None and self.handles.critic_body_name is not None:
            self._install_activation_hooks(
                self.handles.critic_body,
                "critic",
                self.handles.critic_body_name,
            )

    def _capture_normalizer(self, _module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        if not self._capturing:
            return
        if inputs and isinstance(inputs[0], torch.Tensor) and isinstance(output, torch.Tensor):
            self._raw = inputs[0].detach()
            self._normalized = output.detach()

    def _capture_logits(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not self._capturing:
            return
        if isinstance(output, torch.Tensor):
            self._actor_logits = output.detach()

    def _install_activation_hooks(self, body: nn.Module, role: str, body_name: str) -> None:
        for name, module in body.named_modules():
            if not name or not _is_activation(module):
                continue
            label_parts = [role]
            if body_name:
                label_parts.append(body_name)
            label_parts.append(name)
            label = ".".join(label_parts)
            self._hook_handles.append(
                module.register_forward_hook(self._activation_hook(label))
            )

    def _activation_hook(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not self._capturing:
                return
            if isinstance(output, torch.Tensor):
                occurrence = self._activation_counts.get(name, 0)
                label = name if occurrence == 0 else f"{name}#{occurrence}"
                self._activations[label] = output.detach()
                self._activation_counts[name] = occurrence + 1

        return capture

    def _policy_std(self, actor_logits: torch.Tensor | None) -> torch.Tensor | None:
        distribution = self.handles.distribution
        if distribution is None:
            return None
        update = getattr(distribution, "update", None)
        if callable(update) and actor_logits is not None:
            # RSL-RL's deterministic path returns the distribution mean without
            # populating its transient Normal object. Updating from that same mean
            # exposes the actual clamped/transformed std without sampling an action.
            update(actor_logits)
        try:
            value = getattr(distribution, "std", None)
        except (AttributeError, RuntimeError):
            value = None
        if callable(value):
            value = value()
        return value.detach() if isinstance(value, torch.Tensor) else None

    def _std_parameter_count(self) -> int | None:
        distribution = self.handles.distribution
        if distribution is None:
            return None
        parameters = [
            getattr(distribution, name, None)
            for name in ("std_param", "log_std_param")
        ]
        for parameter in parameters:
            if isinstance(parameter, torch.Tensor):
                return int(parameter.numel())
        return None

    def _std_state_dependent(self) -> bool | None:
        distribution = self.handles.distribution
        if distribution is None:
            return None
        return not any(
            isinstance(getattr(distribution, name, None), torch.Tensor)
            for name in ("std_param", "log_std_param")
        )


def _select_observation(observations: Any, groups: tuple[str, ...]) -> torch.Tensor:
    if isinstance(observations, torch.Tensor):
        return observations.detach()
    if isinstance(observations, dict) or hasattr(observations, "__getitem__"):
        tensors = []
        for group in groups:
            try:
                value = observations[group]
            except (KeyError, TypeError, IndexError):
                continue
            if isinstance(value, torch.Tensor):
                tensors.append(value)
        if tensors:
            if len(tensors) == 1:
                return tensors[0].detach()
            return torch.cat(tensors, dim=-1).detach()
    raise ValueError(f"cannot resolve actor observation groups {groups!r} from policy payload")


def _first_environment_vector(value: torch.Tensor) -> torch.Tensor:
    detached = value.detach().to(device="cpu").clone()
    if detached.ndim == 0:
        return detached.reshape(1)
    if detached.ndim >= 2:
        if detached.shape[0] != 1:
            raise ValueError(
                "policy introspection expects a single-environment viewer batch; "
                f"received batch size {detached.shape[0]}"
            )
        detached = detached[0]
    return detached.reshape(-1)


def _is_activation(module: nn.Module) -> bool:
    module_path = type(module).__module__
    return module_path.startswith("torch.nn.modules.activation")


class JacobianCadence:
    """Throttling policy for local raw-observation action sensitivities."""

    SUPPORTED_RATES = (0, 1, 2, 5, 10)

    def __init__(self, frequency_hz: float = 2) -> None:
        frequency = float(frequency_hz)
        if frequency not in self.SUPPORTED_RATES:
            rates = ", ".join(str(rate) for rate in self.SUPPORTED_RATES)
            raise ValueError(f"Jacobian frequency must be one of {rates} Hz")
        self.frequency_hz = frequency
        self._next_due: float | None = None

    def should_calculate(self, *, now: float | None = None) -> bool:
        if self.frequency_hz == 0:
            return False
        current = time.monotonic() if now is None else float(now)
        if self._next_due is not None and current < self._next_due:
            return False
        self._next_due = current + (1.0 / self.frequency_hz)
        return True


def compute_policy_jacobian(
    handles: PolicyRuntimeHandles,
    raw_observation: torch.Tensor,
) -> torch.Tensor:
    """Return ``d deterministic_action / d raw_actor_observation`` on the model device."""

    if not hasattr(torch, "func") or not hasattr(torch.func, "jacrev"):
        raise RuntimeError("this PyTorch build does not provide torch.func.jacrev")
    parameter = next(handles.actor_body.parameters(), None)
    if parameter is None:
        raise ValueError("actor body has no parameters to resolve its device and dtype")
    raw = torch.as_tensor(raw_observation).detach().to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    if raw.ndim == 2 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 1 or raw.numel() != handles.actor_input_dim:
        raise ValueError(
            f"raw actor observation must have shape ({handles.actor_input_dim},), "
            f"got {tuple(raw.shape)}"
        )

    modes = tuple((module, module.training) for module in handles.actor.modules())
    handles.actor.eval()

    def deterministic_action(value: torch.Tensor) -> torch.Tensor:
        batched = value.unsqueeze(0)
        normalizer = handles.actor_normalizer
        model_input = normalizer(batched) if normalizer is not None else batched
        logits = handles.actor_body(model_input)
        distribution = handles.distribution
        deterministic = getattr(distribution, "deterministic_output", None)
        action = deterministic(logits) if callable(deterministic) else logits
        if not isinstance(action, torch.Tensor):
            raise TypeError("deterministic distribution output must be a tensor")
        return action.reshape(-1)

    try:
        with torch.enable_grad():
            jacobian = torch.func.jacrev(deterministic_action)(raw)
    finally:
        for module, training in modes:
            module.training = training
    if jacobian.shape != (handles.action_dim, handles.actor_input_dim):
        raise ValueError(
            "actor Jacobian shape disagrees with the runtime policy contract: "
            f"expected {(handles.action_dim, handles.actor_input_dim)}, "
            f"got {tuple(jacobian.shape)}"
        )
    return jacobian.detach()


def build_action_pipeline_snapshot(
    actor_output: torch.Tensor,
    action_term: Any,
    *,
    wrapper_clip: float | None,
) -> ActionPipelineSnapshot:
    """Read the actual structured-action outputs and mark each clipping stage."""

    raw = _first_environment_vector(actor_output)
    if raw.numel() == 0:
        raise ValueError("actor output cannot be empty")
    if wrapper_clip is not None:
        clip_value = float(wrapper_clip)
        if clip_value <= 0.0:
            raise ValueError("wrapper clip must be positive or None")
        wrapper = torch.clamp(raw, min=-clip_value, max=clip_value)
    else:
        wrapper = raw.clone()

    if isinstance(action_term, StructuredActionStepSnapshot):
        processed = action_term.processed_action
        position_targets = action_term.position_targets
        velocity_targets = action_term.velocity_targets
        leg_slots = action_term.leg_slots
        wheel_slots = action_term.wheel_slots
        names = action_term.channel_names
        joint_limit_clipped = action_term.joint_limit_clipped
    else:
        processed = _first_environment_vector(action_term.raw_action)
        position_targets = _first_environment_vector(action_term.position_targets)
        velocity_targets = _first_environment_vector(action_term.velocity_targets)
        leg_slots = _as_indices(getattr(action_term, "_leg_slots", None))
        wheel_slots = _as_indices(getattr(action_term, "_wheel_slots", None))
        names = getattr(getattr(action_term, "cfg", None), "actuator_names", None)
        if not isinstance(names, (tuple, list)) or len(names) != processed.numel():
            names = JOINT_NAMES if len(JOINT_NAMES) == processed.numel() else tuple(
                f"action_{index}" for index in range(processed.numel())
            )
        joint_limit_clipped = _joint_limit_clipping(
            action_term,
            processed,
            position_targets,
            leg_slots,
        )
    if processed.shape != wrapper.shape:
        raise ValueError(
            "structured action width does not match actor output: "
            f"{processed.numel()} != {wrapper.numel()}"
        )
    if set(leg_slots) & set(wheel_slots) or set(leg_slots) | set(wheel_slots) != set(range(processed.numel())):
        raise ValueError("structured action slots must uniquely partition the action channels")
    if len(names) != processed.numel():
        raise ValueError("action channel names do not match runtime action dimension")
    processed = processed.detach().cpu().clone()
    position_targets = position_targets.detach().cpu().clone()
    velocity_targets = velocity_targets.detach().cpu().clone()
    leg_indices = {slot: index for index, slot in enumerate(leg_slots)}
    wheel_indices = {slot: index for index, slot in enumerate(wheel_slots)}
    targets: list[PhysicalTarget] = []
    for slot, channel in enumerate(names):
        if slot in leg_indices:
            target_index = leg_indices[slot]
            targets.append(
                PhysicalTarget(
                    channel=str(channel),
                    kind="position",
                    value=float(position_targets[target_index].item()),
                    unit="rad",
                    joint_limit_clipped=joint_limit_clipped[target_index],
                )
            )
        else:
            target_index = wheel_indices[slot]
            targets.append(
                PhysicalTarget(
                    channel=str(channel),
                    kind="velocity",
                    value=float(velocity_targets[target_index].item()),
                    unit="rad/s",
                    joint_limit_clipped=False,
                )
            )

    wrapper_flags = tuple(bool(item) for item in torch.ne(raw, wrapper).tolist())
    action_term_flags = tuple(bool(item) for item in torch.ne(wrapper, processed).tolist())
    return ActionPipelineSnapshot(
        actor_output=raw,
        wrapper_clipped=wrapper,
        processed_action=processed,
        wrapper_clipped_flags=wrapper_flags,
        action_term_clipped_flags=action_term_flags,
        targets=tuple(targets),
    )


def _as_indices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, torch.Tensor):
        raise ValueError("structured action term is missing its runtime action slots")
    return tuple(int(item) for item in value.detach().cpu().tolist())


def _joint_limit_clipping(
    action_term: Any,
    processed: torch.Tensor,
    position_targets: torch.Tensor,
    leg_slots: tuple[int, ...],
) -> tuple[bool, ...]:
    try:
        joint_ids = action_term._joint_ids[action_term._leg_slots]
        data = action_term._entity.data
        nominal = data.default_joint_pos[0, joint_ids].detach().cpu()
        requested = nominal + processed[list(leg_slots)] * LEG_POSITION_SCALE_RAD
        return tuple(
            bool(torch.abs(request - target) > 1.0e-6)
            for request, target in zip(requested, position_targets, strict=True)
        )
    except (AttributeError, IndexError, TypeError):
        return tuple(False for _ in leg_slots)

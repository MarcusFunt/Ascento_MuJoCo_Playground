"""On-demand Captum Integrated Gradients for a viewer-owned policy actor."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import torch

from .introspection_contract import PolicyRuntimeHandles


@dataclass(frozen=True)
class IntegratedGradientsResult:
    explanation_id: str
    action_index: int
    action_name: str
    baseline_kind: str
    baseline_description: str
    n_steps: int
    input_raw: list[float]
    baseline_raw: list[float]
    attribution: list[float]
    absolute_fraction: list[float]
    convergence_delta: float
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "status": "complete",
            "method": "integrated_gradients",
            "action_index": self.action_index,
            "action_name": self.action_name,
            "baseline_kind": self.baseline_kind,
            "baseline_description": self.baseline_description,
            "n_steps": self.n_steps,
            "input_raw": self.input_raw,
            "baseline_raw": self.baseline_raw,
            "attribution": self.attribution,
            "absolute_fraction": self.absolute_fraction,
            "convergence_delta": self.convergence_delta,
            "duration_ms": self.duration_ms,
        }


def normalizer_mean_baseline(handles: PolicyRuntimeHandles) -> torch.Tensor:
    """Return the raw observation mapping to zero under the live normalizer."""

    normalizer = handles.actor_normalizer
    if normalizer is None:
        return torch.zeros(handles.actor_input_dim)
    mean = getattr(normalizer, "mean", None)
    if callable(mean):
        mean = mean()
    if not isinstance(mean, torch.Tensor):
        mean = getattr(normalizer, "_mean", None)
    if not isinstance(mean, torch.Tensor):
        raise ValueError("actor normalizer does not expose its running mean for a zero-normalized baseline")
    mean = mean.detach().reshape(-1).cpu().clone()
    if mean.numel() != handles.actor_input_dim or not torch.isfinite(mean).all():
        raise ValueError("actor normalizer mean does not match the runtime observation schema")
    return mean


def explain_integrated_gradients(
    handles: PolicyRuntimeHandles,
    *,
    explanation_id: str,
    input_raw: list[float] | torch.Tensor,
    baseline_raw: list[float] | torch.Tensor,
    action_index: int,
    action_name: str,
    baseline_kind: str,
    baseline_description: str,
    n_steps: int = 32,
) -> IntegratedGradientsResult:
    """Explain one deterministic action output without sampling or mutating policy state."""

    if not 0 <= int(action_index) < handles.action_dim:
        raise ValueError(f"action_index must be in [0, {handles.action_dim - 1}]")
    if not 8 <= int(n_steps) <= 512:
        raise ValueError("Integrated Gradients n_steps must be between 8 and 512")
    module = find_spec("captum")
    if module is None:
        raise RuntimeError(
            "Integrated Gradients requires the optional 'introspection' extra; install with `uv sync --extra introspection`."
        )
    from captum.attr import IntegratedGradients

    parameter = next(handles.actor_body.parameters(), None)
    if parameter is None:
        raise ValueError("actor body has no parameters to resolve its device and dtype")
    input_tensor = torch.as_tensor(input_raw, dtype=parameter.dtype, device=parameter.device).reshape(-1)
    baseline_tensor = torch.as_tensor(baseline_raw, dtype=parameter.dtype, device=parameter.device).reshape(-1)
    if input_tensor.numel() != handles.actor_input_dim or baseline_tensor.numel() != handles.actor_input_dim:
        raise ValueError(
            f"input and baseline must both contain {handles.actor_input_dim} actor observations"
        )
    if not torch.isfinite(input_tensor).all() or not torch.isfinite(baseline_tensor).all():
        raise ValueError("input and baseline observations must be finite")

    def selected_action(raw_batch: torch.Tensor) -> torch.Tensor:
        normalizer = handles.actor_normalizer
        normalized = normalizer(raw_batch) if normalizer is not None else raw_batch
        logits = handles.actor_body(normalized)
        deterministic = getattr(handles.distribution, "deterministic_output", None)
        actions = deterministic(logits) if callable(deterministic) else logits
        return actions[..., int(action_index)]

    modes = tuple((child, child.training) for child in handles.actor.modules())
    handles.actor.eval()
    started = time.perf_counter()
    try:
        with torch.enable_grad():
            integrated_gradients = IntegratedGradients(selected_action)
            attributions, delta = integrated_gradients.attribute(
                input_tensor.unsqueeze(0),
                baselines=baseline_tensor.unsqueeze(0),
                n_steps=int(n_steps),
                return_convergence_delta=True,
            )
    finally:
        for child, training in modes:
            child.training = training
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    attribution = attributions.detach().reshape(-1).cpu()
    if attribution.numel() != handles.actor_input_dim or not torch.isfinite(attribution).all():
        raise ValueError("Integrated Gradients returned an invalid attribution vector")
    absolute = attribution.abs()
    total = float(absolute.sum().item())
    fractions = absolute / total if total > 0 else torch.zeros_like(absolute)
    delta_value = float(delta.detach().reshape(-1)[0].cpu().item())
    if not math.isfinite(delta_value):
        raise ValueError("Integrated Gradients returned a non-finite convergence delta")
    return IntegratedGradientsResult(
        explanation_id=str(explanation_id),
        action_index=int(action_index),
        action_name=str(action_name),
        baseline_kind=str(baseline_kind),
        baseline_description=str(baseline_description),
        n_steps=int(n_steps),
        input_raw=input_tensor.detach().cpu().tolist(),
        baseline_raw=baseline_tensor.detach().cpu().tolist(),
        attribution=attribution.tolist(),
        absolute_fraction=fractions.tolist(),
        convergence_delta=delta_value,
        duration_ms=elapsed_ms,
    )

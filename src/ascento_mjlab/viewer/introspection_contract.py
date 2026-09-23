"""Runtime handles for the viewer-only policy introspection path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyRuntimeHandles:
    """Resolved model pieces used by diagnostics, never by the trainer."""

    actor: nn.Module
    critic: nn.Module | None
    actor_normalizer: nn.Module | None
    critic_normalizer: nn.Module | None
    distribution: nn.Module | None
    actor_body: nn.Module
    critic_body: nn.Module | None
    actor_body_name: str
    critic_body_name: str | None
    actor_obs_groups: tuple[str, ...]
    critic_obs_groups: tuple[str, ...]
    actor_input_dim: int
    critic_input_dim: int | None
    action_dim: int


def resolve_runtime_handles(runner: Any) -> PolicyRuntimeHandles:
    """Resolve and validate the concrete RSL-RL actor/critic runtime contract.

    This deliberately refuses to guess the tensor topology from a checkpoint's
    serialized width. The live modules determine dimensions and observation
    groups, while the environment manager supplies the semantic feature schema.
    """

    algorithm = getattr(runner, "alg", None)
    get_policy = getattr(algorithm, "get_policy", None)
    if not callable(get_policy):
        raise ValueError("runner does not expose alg.get_policy()")
    actor = get_policy()
    if not isinstance(actor, nn.Module):
        raise ValueError("runner policy is not a torch module")

    critic = getattr(algorithm, "critic", None)
    if critic is not None and not isinstance(critic, nn.Module):
        raise ValueError("runner critic is not a torch module")

    actor_body_name, actor_body = _resolve_body(actor)
    actor_input_dim, action_dim = _linear_dimensions(actor_body, "actor")
    actor_groups = _resolve_groups(actor, "actor")

    critic_body_name: str | None = None
    critic_input_dim: int | None = None
    critic_groups: tuple[str, ...] = ()
    critic_body: nn.Module | None = None
    if critic is not None:
        critic_body_name, critic_body = _resolve_body(critic)
        critic_input_dim, _critic_output_dim = _linear_dimensions(critic_body, "critic")
        critic_groups = _resolve_groups(critic, "critic")

    return PolicyRuntimeHandles(
        actor=actor,
        critic=critic,
        actor_normalizer=_optional_module(actor, "obs_normalizer"),
        critic_normalizer=(
            _optional_module(critic, "obs_normalizer") if critic is not None else None
        ),
        distribution=_optional_module(actor, "distribution"),
        actor_body=actor_body,
        critic_body=critic_body,
        actor_body_name=actor_body_name,
        critic_body_name=critic_body_name,
        actor_obs_groups=actor_groups,
        critic_obs_groups=critic_groups,
        actor_input_dim=actor_input_dim,
        critic_input_dim=critic_input_dim,
        action_dim=action_dim,
    )


def build_policy_branch_schema(
    model: nn.Module,
    body: nn.Module,
    *,
    distribution: nn.Module | None = None,
) -> dict[str, Any]:
    """Describe the active branch's architecture using checkpoint-owned modules."""

    linears = [
        (name, module)
        for name, module in body.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not linears:
        raise ValueError("policy branch has no linear layers")
    linear_stats: list[dict[str, Any]] = []
    for index, (name, layer) in enumerate(linears):
        weight = layer.weight.detach().float().cpu()
        bias = layer.bias.detach().float().cpu() if layer.bias is not None else None
        linear_stats.append(
            {
                "module_index": index,
                "name": name,
                "input_size": int(layer.in_features),
                "output_size": int(layer.out_features),
                "weight_count": int(weight.numel()),
                "weight_rms": float(torch.mean(weight.square()).sqrt().item()),
                "weight_mean_abs": float(weight.abs().mean().item()),
                "weight_max_abs": float(weight.abs().max().item()),
                "bias_rms": float(torch.mean(bias.square()).sqrt().item()) if bias is not None else None,
            }
        )
    activation = next(
        (
            type(module).__name__.upper()
            for _name, module in body.named_modules()
            if _is_activation(module)
        ),
        "unknown",
    )
    dist_name = type(distribution).__name__ if distribution is not None else None
    if dist_name and "gaussian" in dist_name.lower():
        dist_name = "Gaussian"
    std_parameters = None
    if distribution is not None:
        for name in ("std_param", "log_std_param"):
            value = getattr(distribution, name, None)
            if isinstance(value, torch.Tensor):
                std_parameters = int(value.numel())
                break
    return {
        "layers": [int(linears[0][1].in_features)]
        + [int(layer.out_features) for _name, layer in linears],
        "activation": activation,
        "distribution": dist_name,
        "std_parameters": std_parameters,
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "linear_layers": linear_stats,
    }


def _is_activation(module: nn.Module) -> bool:
    return type(module).__module__.startswith("torch.nn.modules.activation")


def _optional_module(owner: nn.Module | None, name: str) -> nn.Module | None:
    value = getattr(owner, name, None) if owner is not None else None
    if value is None:
        return None
    if not isinstance(value, nn.Module):
        raise ValueError(f"policy {name} is not a torch module")
    return value


def _resolve_body(model: nn.Module) -> tuple[str, nn.Module]:
    for name in ("mlp", "network", "actor", "critic"):
        candidate = getattr(model, name, None)
        if isinstance(candidate, nn.Module) and _linear_layers(candidate):
            return name, candidate
    if _linear_layers(model):
        return "", model
    raise ValueError(f"{type(model).__name__} has no introspectable linear policy body")


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [child for child in module.modules() if isinstance(child, nn.Linear)]


def _linear_dimensions(module: nn.Module, label: str) -> tuple[int, int]:
    layers = _linear_layers(module)
    if not layers:
        raise ValueError(f"{label} body has no linear layers")
    return int(layers[0].in_features), int(layers[-1].out_features)


def _resolve_groups(model: nn.Module, role: str) -> tuple[str, ...]:
    groups = getattr(model, "obs_groups", None)
    if isinstance(groups, dict):
        selected = groups.get(role)
        if isinstance(selected, str):
            selected = (selected,)
        if isinstance(selected, (tuple, list)) and selected:
            if not all(isinstance(item, str) and item for item in selected):
                raise ValueError(f"{role} observation group names must be nonempty strings")
            return tuple(selected)
    elif isinstance(groups, str):
        return (groups,)
    elif isinstance(groups, (tuple, list)) and groups:
        if not all(isinstance(item, str) and item for item in groups):
            raise ValueError(f"{role} observation group names must be nonempty strings")
        return tuple(groups)
    # Simple one-group policies in tests and third-party runners often omit
    # obs_groups. Keep the fallback explicit and role-scoped.
    return (role,)

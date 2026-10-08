"""Semantic actor/critic observation normalization with exact checkpoint migration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.normalization import EmpiricalNormalization

_EPSILON = 1.0e-2
_ACTOR_DIM = 41
_CRITIC_DIM = 50
_RUNNING_PROPRIO_DIM = 22
_TARGET_XY = (38, 39)
_TARGET_HEADING = 40
_PI = 3.141592653589793
_MODE_INDEX = 41

_CONTRACT: dict[str, Any] = {
    "schema": "ascento_semantic_normalization/v1",
    "implementation": "rsl_rl.EmpiricalNormalization.selective_v1",
    "epsilon": _EPSILON,
    "actor_observation_dim": _ACTOR_DIM,
    "critic_observation_dim": _CRITIC_DIM,
    "running_normalized": {
        "gravity_xyz": [0, 1, 2],
        "base_lin_vel_xyz": [3, 4, 5],
        "base_ang_vel_xyz": [6, 7, 8],
        "base_height_m": [9],
        "joint_pos_rad": [10, 11, 12, 13, 14, 15],
        "joint_vel_rad_s": [16, 17, 18, 19, 20, 21],
    },
    "fixed_scaled": {
        "world_target_error_xy_m": {"indices": [38, 39], "center": [0.0, 0.0], "scale": [1.0, 1.0]},
        "world_target_heading_error_rad": {
            "indices": [_TARGET_HEADING],
            "center": [0.0],
            "scale": [_PI],
        },
    },
    "identity_or_fixed": {
        "wheel_contact_booleans": {"indices": [22, 23], "scale": 1.0},
        "wheel_contact_forces_100n": {"indices": [24, 25], "scale": 1.0},
        "normalized_actuator_effort": {"indices": [26, 27, 28, 29, 30, 31], "scale": 1.0},
        "previous_normalized_action": {"indices": [32, 33, 34, 35, 36, 37], "scale": 1.0},
    },
    "critic_extra_running_channels": [
        "root_pos_xyz",
        "root_lin_vel_w_xyz",
        "root_ang_vel_w_xyz",
    ],
}

_CONTEXT_CONTRACT: dict[str, Any] = {
    **_CONTRACT,
    "schema": "ascento_semantic_normalization/v2",
    "actor_observation_dim": 42,
    "critic_observation_dim": 51,
    "identity_or_fixed": {
        **_CONTRACT["identity_or_fixed"],
        "obstacle_mode": {"index": _MODE_INDEX, "scale": 1.0},
    },
}


def semantic_normalizer_contract() -> dict[str, Any]:
    """Return the JSON-serializable normalization ABI recorded in checkpoints."""
    return deepcopy(_CONTRACT)


def context_mode_normalizer_contract() -> dict[str, Any]:
    """Return the semantic-normalization ABI for the 42D context actor."""
    return deepcopy(_CONTEXT_CONTRACT)


def _normalization_layout(obs_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if obs_dim not in (_ACTOR_DIM, _CRITIC_DIM, 42, 51):
        raise ValueError(
            f"semantic normalization supports 41D/42D actor or 50D/51D critic input, got {obs_dim}"
        )
    running = torch.zeros(obs_dim, dtype=torch.bool)
    running[:_RUNNING_PROPRIO_DIM] = True
    if obs_dim in (_CRITIC_DIM, 51):
        running[42 if obs_dim == 51 else _ACTOR_DIM :] = True
    center = torch.zeros(obs_dim, dtype=torch.float32)
    scale = torch.ones(obs_dim, dtype=torch.float32)
    scale[_TARGET_HEADING] = _PI
    return running, center, scale


class SemanticEmpiricalNormalization(EmpiricalNormalization):
    """Use running statistics for proprioception and fixed units for commands."""

    def __init__(self, shape: int | tuple[int, ...] | list[int]) -> None:
        if isinstance(shape, int):
            obs_dim = shape
        elif isinstance(shape, (tuple, list)) and len(shape) == 1:
            obs_dim = int(shape[0])
        else:
            raise ValueError(
                f"semantic normalization expects a one-dimensional input, got {shape!r}"
            )
        running, center, scale = _normalization_layout(obs_dim)
        super().__init__(obs_dim, eps=_EPSILON)
        self.register_buffer("running_mask", running.unsqueeze(0), persistent=False)
        self.register_buffer("fixed_center", center.unsqueeze(0), persistent=False)
        self.register_buffer("fixed_scale", scale.unsqueeze(0), persistent=False)
        fixed = ~running
        with torch.no_grad():
            self._mean[:, fixed] = self.fixed_center[:, fixed]
            self._var[:, fixed] = self.fixed_scale[:, fixed].square()
            self._std[:, fixed] = self.fixed_scale[:, fixed]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.running_mask.shape[-1]:
            raise ValueError(
                f"semantic normalizer expects {self.running_mask.shape[-1]} channels, got {x.shape[-1]}"
            )
        running = (x - self._mean) / (self._std + self.eps)
        fixed = (x - self.fixed_center) / self.fixed_scale
        return torch.where(self.running_mask, running, fixed)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Update only running-normalized channels using RSL-RL's empirical rule."""
        if not self.training or (self.until is not None and self.count >= self.until):
            return
        if x.ndim != 2 or x.shape[-1] != self.running_mask.shape[-1]:
            raise ValueError(
                f"semantic normalizer update expects (batch, {self.running_mask.shape[-1]}), got {tuple(x.shape)}"
            )
        selected = self.running_mask.reshape(-1)
        running_x = x[:, selected]
        count_x = running_x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(running_x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(running_x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean[:, selected]
        new_mean = self._mean[:, selected] + rate * delta_mean
        new_var = self._var[:, selected] + rate * (
            var_x - self._var[:, selected] + delta_mean * (mean_x - new_mean)
        )
        self._mean[:, selected] = new_mean
        self._var[:, selected] = new_var
        self._std[:, selected] = torch.sqrt(new_var)


class SemanticNormalizationMLP(MLPModel):
    """RSL-RL MLP whose fixed command scales cannot drift with curriculum."""

    normalizer_contract = staticmethod(semantic_normalizer_contract)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.obs_normalization:
            raise ValueError("semantic normalization requires obs_normalization=True")
        self.obs_normalizer = SemanticEmpiricalNormalization(self.obs_dim)


def migrate_semantic_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, epsilon: float = _EPSILON
) -> dict[str, torch.Tensor]:
    """Reparameterize the first MLP layer for semantic fixed command scales.

    The source implementation computes ``(x - mean) / (std + epsilon)``.
    For the fixed channels, this transforms the first-layer weights and bias so
    that the physical-input-to-preactivation mapping is unchanged.
    """
    if abs(float(epsilon) - _EPSILON) > 1.0e-12:
        raise ValueError(f"pinned RSL-RL normalization epsilon is {_EPSILON}, got {epsilon}")
    required = (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "mlp.0.weight",
        "mlp.0.bias",
    )
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise ValueError(f"state dict is missing semantic migration tensors: {missing}")
    result = {name: value.detach().clone() for name, value in state_dict.items()}
    mean = result["obs_normalizer._mean"].reshape(-1)
    std = result["obs_normalizer._std"].reshape(-1)
    obs_dim = mean.numel()
    running, center, scale = _normalization_layout(obs_dim)
    old_weight = result["mlp.0.weight"].clone()
    old_bias = result["mlp.0.bias"].clone()
    fixed = ~running
    effective_old_scale = std + _EPSILON
    result["mlp.0.weight"][:, fixed] = (
        old_weight[:, fixed]
        * scale[fixed].to(old_weight)
        / effective_old_scale[fixed].to(old_weight)
    )
    correction = (center[fixed].to(old_weight) - mean[fixed].to(old_weight)) / effective_old_scale[
        fixed
    ].to(old_weight)
    result["mlp.0.bias"] = old_bias + torch.sum(
        old_weight[:, fixed] * correction.unsqueeze(0), dim=1
    )

    fixed_full = fixed.reshape(1, -1)
    center_full = center.reshape(1, -1).to(result["obs_normalizer._mean"])
    scale_full = scale.reshape(1, -1).to(result["obs_normalizer._std"])
    result["obs_normalizer._mean"][fixed_full] = center_full[fixed_full]
    result["obs_normalizer._var"][fixed_full] = scale_full[fixed_full].square()
    result["obs_normalizer._std"][fixed_full] = scale_full[fixed_full]
    return result


def semantic_model_class_name() -> str:
    """Return the class path understood by RSL-RL's model resolver."""
    return f"{__name__}:SemanticNormalizationMLP"


def model_normalizer_contract(model: Any) -> dict[str, Any]:
    """Return the normalization contract implemented by an RSL-RL model."""
    if isinstance(getattr(model, "obs_normalizer", None), SemanticEmpiricalNormalization):
        if int(model.obs_dim) in (42, 51):
            return context_mode_normalizer_contract()
        return semantic_normalizer_contract()
    contract_factory = getattr(model, "normalizer_contract", None)
    if callable(contract_factory):
        return contract_factory()
    return {"schema": "ascento_empirical_normalization/v1", "epsilon": _EPSILON}


def configure_model_normalizer_for_contract(model: Any, contract: Mapping[str, Any] | None) -> None:
    """Select checkpoint normalization semantics on an already-created model."""
    if not hasattr(model, "obs_dim") or not hasattr(model, "obs_normalizer"):
        raise TypeError("active model does not expose RSL-RL observation normalization")
    schema = contract.get("schema") if isinstance(contract, Mapping) else None
    if schema in (_CONTRACT["schema"], _CONTEXT_CONTRACT["schema"]):
        expected_dims = (41, 50) if schema == _CONTRACT["schema"] else (42, 51)
        if int(model.obs_dim) not in expected_dims:
            raise ValueError(
                f"checkpoint normalizer contract {schema!r} does not match "
                f"the active {int(model.obs_dim)}D model"
            )
        model.obs_normalizer = SemanticEmpiricalNormalization(int(model.obs_dim))
        return
    if schema in (None, "ascento_empirical_normalization/v1", "rsl_rl_empirical_normalization/v1"):
        if isinstance(model.obs_normalizer, SemanticEmpiricalNormalization):
            model.obs_normalizer = EmpiricalNormalization(int(model.obs_dim))
        return
    raise ValueError(f"unsupported checkpoint normalizer contract schema {schema!r}")


def configure_rl_cfg_for_normalizer_contract(
    agent_cfg: Any, contract: Mapping[str, Any] | None
) -> Any:
    """Select the model implementation that matches checkpoint normalization metadata."""
    schema = contract.get("schema") if isinstance(contract, Mapping) else None
    if schema in (_CONTRACT["schema"], _CONTEXT_CONTRACT["schema"]):
        expected_dims = (_ACTOR_DIM, _CRITIC_DIM) if schema == _CONTRACT["schema"] else (42, 51)
        declared_dims = (
            (
                contract.get("actor_observation_dim"),
                contract.get("critic_observation_dim"),
            )
            if isinstance(contract, Mapping)
            else (None, None)
        )
        if declared_dims != expected_dims:
            raise ValueError(
                f"invalid semantic normalizer dimensions in contract: {declared_dims!r}"
            )
        class_name = semantic_model_class_name()
        agent_cfg.actor.class_name = class_name
        agent_cfg.critic.class_name = class_name
        return agent_cfg
    if schema in (None, "ascento_empirical_normalization/v1", "rsl_rl_empirical_normalization/v1"):
        agent_cfg.actor.class_name = "MLPModel"
        agent_cfg.critic.class_name = "MLPModel"
        return agent_cfg
    raise ValueError(f"unsupported checkpoint normalizer contract schema {schema!r}")


def require_model_normalizer_contract(infos: Mapping[str, Any] | None, model: Any) -> None:
    """Reject checkpoints loaded into a model with different normalization semantics."""
    stored = infos.get("normalizer_contract") if isinstance(infos, Mapping) else None
    expected = model_normalizer_contract(model)
    if stored is None:
        if expected.get("schema") == "ascento_empirical_normalization/v1":
            return
        raise ValueError("checkpoint lacks the semantic normalizer contract required by this model")
    if stored != expected:
        raise ValueError(
            "checkpoint normalizer contract does not match the active actor/critic model"
        )


def migrate_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate an iteration-zero checkpoint payload without touching optimizer state."""
    if int(payload.get("iter", -1)) != 0:
        raise ValueError("semantic migration requires an iteration-zero checkpoint")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, Mapping) or optimizer.get("state"):
        raise ValueError("semantic migration requires an empty optimizer state")
    infos = payload.get("infos")
    if not isinstance(infos, Mapping):
        raise ValueError("checkpoint lacks provenance infos")
    if infos.get("normalizer_contract") is not None:
        raise ValueError("checkpoint already declares a normalizer contract")
    actor = payload.get("actor_state_dict")
    critic = payload.get("critic_state_dict")
    if not isinstance(actor, Mapping) or not isinstance(critic, Mapping):
        raise ValueError("checkpoint must contain actor_state_dict and critic_state_dict")
    result = dict(payload)
    result["actor_state_dict"] = migrate_semantic_state_dict(actor)
    result["critic_state_dict"] = migrate_semantic_state_dict(critic)
    result["infos"] = {**dict(infos), "normalizer_contract": semantic_normalizer_contract()}
    return result


def _insert_mode_channel(
    state_dict: Mapping[str, torch.Tensor], *, index: int
) -> dict[str, torch.Tensor]:
    required = (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "mlp.0.weight",
    )
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise ValueError(f"state dict is missing context migration tensors: {missing}")
    result = {name: value.detach().clone() for name, value in state_dict.items()}
    old_weight = result["mlp.0.weight"]
    if old_weight.ndim != 2 or old_weight.shape[1] not in (41, 50):
        raise ValueError(
            f"context migration requires a 41D actor or 50D critic, got {old_weight.shape}"
        )
    if index != _MODE_INDEX or index > old_weight.shape[1]:
        raise ValueError(f"invalid obstacle_mode insertion index {index}")
    result["mlp.0.weight"] = torch.cat(
        (
            old_weight[:, :index],
            old_weight.new_zeros((old_weight.shape[0], 1)),
            old_weight[:, index:],
        ),
        dim=1,
    )
    for key, fill_value in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        value = result[key]
        if value.shape[-1] != old_weight.shape[1]:
            raise ValueError(f"{key} width does not match the first layer input width")
        mode_value = torch.full_like(value[..., :1], fill_value)
        result[key] = torch.cat((value[..., :index], mode_value, value[..., index:]), dim=-1)
    return result


def migrate_context_mode_checkpoint_payload(
    payload: Mapping[str, Any], *, target_task_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Insert a zero-initialized obstacle-mode input into a semantic v1 checkpoint."""
    if int(payload.get("iter", -1)) != 0:
        raise ValueError("context-mode migration requires an iteration-zero checkpoint")
    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, Mapping) or optimizer.get("state"):
        raise ValueError("context-mode migration requires an empty optimizer state")
    infos = payload.get("infos")
    if not isinstance(infos, Mapping):
        raise ValueError("checkpoint lacks provenance infos")
    if infos.get("normalizer_contract") != semantic_normalizer_contract():
        raise ValueError("context-mode migration requires the semantic-normalization v1 contract")
    if not isinstance(infos.get("task_contract"), Mapping):
        raise ValueError("checkpoint lacks its source task contract")
    if not isinstance(target_task_contract, Mapping):
        raise ValueError("target task contract is required for context-mode migration")
    source_topology = infos["task_contract"].get("topology")
    target_topology = target_task_contract.get("topology")
    if not isinstance(source_topology, Mapping) or not isinstance(target_topology, Mapping):
        raise ValueError("source and target task contracts must contain signed topologies")
    if source_topology.get("task_id") != target_topology.get("task_id"):
        raise ValueError("context-mode migration requires the same source and target task identity")
    source_observations = source_topology.get("observations")
    target_observations = target_topology.get("observations")
    if not isinstance(source_observations, Mapping) or not isinstance(target_observations, Mapping):
        raise ValueError("source and target task contracts must contain observation groups")
    for group_name in ("actor", "critic"):
        source_group = source_observations.get(group_name)
        target_group = target_observations.get(group_name)
        source_terms = source_group.get("terms") if isinstance(source_group, Mapping) else None
        target_terms = target_group.get("terms") if isinstance(target_group, Mapping) else None
        if not isinstance(source_terms, Mapping) or not isinstance(target_terms, Mapping):
            raise ValueError(
                f"source and target task contracts lack {group_name} observation terms"
            )
        if "obstacle_mode" in source_terms or "obstacle_mode" not in target_terms:
            raise ValueError(
                f"{group_name} task contract must migrate from no obstacle_mode term to one explicit term"
            )
    actor = payload.get("actor_state_dict")
    critic = payload.get("critic_state_dict")
    if not isinstance(actor, Mapping) or not isinstance(critic, Mapping):
        raise ValueError("checkpoint must contain actor_state_dict and critic_state_dict")
    if actor.get("mlp.0.weight", torch.empty(0)).shape[-1:] != (41,):
        raise ValueError("context-mode migration requires a 41D actor input")
    if critic.get("mlp.0.weight", torch.empty(0)).shape[-1:] != (50,):
        raise ValueError("context-mode migration requires a 50D critic input")
    result = dict(payload)
    result["actor_state_dict"] = _insert_mode_channel(actor, index=_MODE_INDEX)
    result["critic_state_dict"] = _insert_mode_channel(critic, index=_MODE_INDEX)
    result["infos"] = {
        **dict(infos),
        "normalizer_contract": context_mode_normalizer_contract(),
        "task_contract": deepcopy(dict(target_task_contract)),
    }
    return result


__all__ = [
    "SemanticEmpiricalNormalization",
    "SemanticNormalizationMLP",
    "configure_rl_cfg_for_normalizer_contract",
    "configure_model_normalizer_for_contract",
    "context_mode_normalizer_contract",
    "migrate_checkpoint_payload",
    "migrate_context_mode_checkpoint_payload",
    "migrate_semantic_state_dict",
    "model_normalizer_contract",
    "require_model_normalizer_contract",
    "semantic_model_class_name",
    "semantic_normalizer_contract",
]

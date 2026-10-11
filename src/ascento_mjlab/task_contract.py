"""Immutable topology contract for an Ascento task configuration.

The plant contract answers whether the robot and simulator match, while the
action contract answers whether the policy's six values mean the same thing.
Neither protects a checkpoint when an observation, reward, command, reset, or
termination topology changes.  This module fingerprints precisely those
policy-facing task semantics without treating operational batch size or the
adaptive episode horizon as part of the ABI.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .physics import REWARD_SCHEMA_VERSION

TASK_CONTRACT_SCHEMA_VERSION = 1
TASK_CONTRACT_ID = "ascento_task_topology_v1"


class TaskContractStatus(StrEnum):
    """Compatibility states emitted for task-topology checkpoint checks."""

    CURRENT = "current"
    MIGRATED_V1_NULL_TASK_ID = "migrated_v1_null_task_id"
    LEGACY = "legacy"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class TaskContractCompatibility:
    """A machine-readable decision about a checkpoint task contract."""

    status: TaskContractStatus
    reason: str
    checkpoint_topology_sha256: str | None
    current_topology_sha256: str | None

    @property
    def is_compatible(self) -> bool:
        return self.status in {
            TaskContractStatus.CURRENT,
            TaskContractStatus.MIGRATED_V1_NULL_TASK_ID,
        }

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "status": self.status.value,
            "compatible": self.is_compatible,
            "reason": self.reason,
            "checkpoint_topology_sha256": self.checkpoint_topology_sha256,
            "current_topology_sha256": self.current_topology_sha256,
        }


@dataclass(frozen=True)
class ActorTransferCompatibility:
    """Decision for an explicit actor-only initialization across task ABIs.

    This is intentionally distinct from :class:`TaskContractCompatibility`.
    A normal resume requires an exact task topology; actor transfer only admits
    a verified match of the actor observations, action semantics, and control
    timestep, and never carries the critic, optimizer, iteration, or event
    state into the target task.
    """

    compatible: bool
    reason: str
    source_actor_abi_sha256: str | None
    target_actor_abi_sha256: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "compatible": self.compatible,
            "reason": self.reason,
            "source_actor_abi_sha256": self.source_actor_abi_sha256,
            "target_actor_abi_sha256": self.target_actor_abi_sha256,
        }


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", value.__class__.__module__)
    qualname = getattr(value, "__qualname__", value.__class__.__qualname__)
    return f"{module}.{qualname}"


def _canonical(value: Any) -> Any:
    """Convert manager configuration into JSON-stable, semantic-only data."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("task configuration contains a non-finite numeric value")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, slice):
        return {"slice": [_canonical(value.start), _canonical(value.stop), _canonical(value.step)]}
    if callable(value):
        return {"callable": _callable_name(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical(item) for item in value), key=repr)
    raise TypeError(f"task configuration value is not contract-serializable: {type(value)!r}")


def _terms(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(name): _canonical(term) for name, term in (value or {}).items()}


def task_topology(cfg: Any) -> dict[str, Any]:
    """Return the stable task semantics represented by an environment config."""
    observations = getattr(cfg, "observations", {}) or {}
    return {
        "task_id": getattr(cfg, "task_id", None),
        "decimation": int(cfg.decimation),
        "scale_rewards_by_dt": bool(getattr(cfg, "scale_rewards_by_dt", False)),
        "observations": {str(name): _canonical(group) for name, group in observations.items()},
        "actions": _terms(getattr(cfg, "actions", None)),
        "commands": _terms(getattr(cfg, "commands", None)),
        "rewards": _terms(getattr(cfg, "rewards", None)),
        "terminations": _terms(getattr(cfg, "terminations", None)),
        "events": _terms(getattr(cfg, "events", None)),
        "metrics": _terms(getattr(cfg, "metrics", None)),
    }


def current_task_contract(cfg: Any) -> dict[str, Any]:
    """Return the exact task ABI required to resume or roll out a policy."""
    topology = task_topology(cfg)
    encoded = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "id": TASK_CONTRACT_ID,
        "reward_schema": REWARD_SCHEMA_VERSION,
        "topology_sha256": hashlib.sha256(encoded).hexdigest(),
        "topology": topology,
    }


def current_task_contract_for_task(task: str, *, play: bool = False) -> dict[str, Any]:
    """Load a registered task lazily so dashboard metadata remains import-safe."""
    from mjlab.tasks.registry import load_env_cfg

    import ascento_mjlab.tasks  # noqa: F401

    cfg = load_env_cfg(task, play=play)
    if task == "Ascento-Locomotion-Speed-Flat":
        from .tasks.locomotion_speed.env_cfg import configure_speed_command_cap

        cfg = configure_speed_command_cap(cfg)
    return current_task_contract(cfg)


def canonical_task_cfg_for_runtime_cfg(cfg: Any) -> Any:
    """Recover one registered task config when serialization dropped ``task_id``.

    ``task_id`` is intentionally a dynamic attribute on mjlab's environment
    configuration, so third-party dataclass serialization can omit it before a
    training runner is constructed.  Exact resume must still validate against
    the registered training ABI, not against a weakened no-ID runtime object.
    Infer the task only when removing the task ID makes one and only one
    registered topology exactly match the runtime topology.
    """
    task_id = getattr(cfg, "task_id", None)
    if isinstance(task_id, str) and task_id:
        from mjlab.tasks.registry import load_env_cfg

        return load_env_cfg(task_id, play=False)

    runtime_topology = task_topology(cfg)
    matches: list[Any] = []
    # This import is delayed to avoid the registry/provenance import cycle.
    from mjlab.tasks.registry import load_env_cfg

    from .tasks import ASCENTO_TASK_IDS

    for candidate_id in ASCENTO_TASK_IDS:
        candidate = load_env_cfg(candidate_id, play=False)
        candidate_topology = task_topology(candidate)
        candidate_topology["task_id"] = None
        if candidate_topology == runtime_topology:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else cfg


def _contract_sha256(contract: Mapping[str, Any] | None) -> str | None:
    if not isinstance(contract, Mapping):
        return None
    value = contract.get("topology_sha256")
    return value if isinstance(value, str) and value else None


def _topology_sha256(topology: Mapping[str, Any]) -> str:
    encoded = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _actor_transfer_topology(contract: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Extract the immutable subset that permits actor/normalizer transfer."""
    if not isinstance(contract, Mapping):
        return None
    topology = contract.get("topology")
    if (
        contract.get("schema_version") != TASK_CONTRACT_SCHEMA_VERSION
        or contract.get("id") != TASK_CONTRACT_ID
        or not isinstance(topology, Mapping)
        or not isinstance(topology.get("observations"), Mapping)
        or "actor" not in topology["observations"]
        or not isinstance(topology.get("actions"), Mapping)
    ):
        return None
    return {
        "decimation": topology.get("decimation"),
        "scale_rewards_by_dt": topology.get("scale_rewards_by_dt"),
        "actor_observations": topology["observations"]["actor"],
        "actions": topology["actions"],
    }


def classify_actor_transfer_compatibility(
    source_contract: Mapping[str, Any] | None,
    target_contract: Mapping[str, Any] | None,
) -> ActorTransferCompatibility:
    """Require a signed actor ABI match before initializing a new task.

    Reward, reset/event, termination, critic-observation, and task-ID changes
    are deliberately allowed here because this API creates a fresh lineage,
    optimizer, critic, and training iteration.  All normal resumes continue to
    use the exact task-contract checker above.
    """
    source = _actor_transfer_topology(source_contract)
    target = _actor_transfer_topology(target_contract)
    source_sha = _topology_sha256(source) if source is not None else None
    target_sha = _topology_sha256(target) if target is not None else None
    if source is None:
        return ActorTransferCompatibility(
            False,
            "source checkpoint lacks a signed actor-observation/action ABI",
            source_sha,
            target_sha,
        )
    if target is None:
        return ActorTransferCompatibility(
            False,
            "target task lacks a signed actor-observation/action ABI",
            source_sha,
            target_sha,
        )
    if source != target:
        return ActorTransferCompatibility(
            False,
            "actor observations, action semantics, control timestep, or reward-dt scaling differ",
            source_sha,
            target_sha,
        )
    return ActorTransferCompatibility(
        True,
        "actor observations, actions, and control timestep exactly match",
        source_sha,
        target_sha,
    )


def classify_zero_obstacle_mode_actor_extension(
    source_contract: Mapping[str, Any] | None,
    target_contract: Mapping[str, Any] | None,
    *,
    source_reference_contract: Mapping[str, Any] | None,
    source_actor_term_order: Sequence[str],
    target_actor_term_order: Sequence[str],
    obstacle_mode: int,
) -> ActorTransferCompatibility:
    """Allow a policy-preserving locomotion-to-Generalist mode-zero lift.

    The source must match the registered flat-locomotion actor ABI. The
    Generalist actor may append exactly its fixed ``obstacle_mode`` scalar,
    with that scalar configured to zero. Callers must then append a zero
    first-layer weight and a neutral normalizer channel before loading.
    """
    exact = classify_actor_transfer_compatibility(source_contract, target_contract)

    def reject(reason: str) -> ActorTransferCompatibility:
        return ActorTransferCompatibility(
            False,
            reason,
            exact.source_actor_abi_sha256,
            exact.target_actor_abi_sha256,
        )

    if obstacle_mode != 0:
        return reject("zero-mode actor extension requires obstacle mode 0")
    if not isinstance(source_contract, Mapping) or not isinstance(target_contract, Mapping):
        return reject("source and target actor task contracts are required")
    if not isinstance(source_reference_contract, Mapping):
        return reject("registered locomotion actor contract is required")

    source_reference_topology = source_reference_contract.get("topology")
    target_task_topology = target_contract.get("topology")
    if not isinstance(source_reference_topology, Mapping) or not isinstance(
        target_task_topology, Mapping
    ):
        return reject("source and target task contracts must contain signed topologies")
    source_reference_task = source_reference_topology.get("task_id")
    target_task = target_task_topology.get("task_id")
    if source_reference_task != "Ascento-Locomotion-Flat":
        return reject("zero-mode extension is limited to the registered flat-locomotion actor")
    if target_task != "Ascento-Generalist-Locomotion-Flat":
        return reject("zero-mode extension is limited to Generalist flat locomotion")

    source_matches_reference = classify_actor_transfer_compatibility(
        source_contract, source_reference_contract
    )
    if not source_matches_reference.compatible:
        return reject("source actor does not exactly match the registered locomotion ABI")

    target_topology = target_contract.get("topology")
    if not isinstance(target_topology, Mapping):
        return reject("target task contract lacks a signed topology")
    observations = target_topology.get("observations")
    target_actor_group = observations.get("actor") if isinstance(observations, Mapping) else None
    target_terms = (
        target_actor_group.get("terms") if isinstance(target_actor_group, Mapping) else None
    )
    if not isinstance(target_terms, Mapping) or "obstacle_mode" not in target_terms:
        return reject("target actor contract lacks the obstacle_mode observation")

    expected_mode_term = {
        "func": {"callable": "ascento_mjlab.mdp.observations.obstacle_mode"},
        "params": {},
        "noise": None,
        "clip": None,
        "scale": None,
        "delay_min_lag": 0,
        "delay_max_lag": 0,
        "delay_per_env": True,
        "delay_hold_prob": 0.0,
        "delay_update_period": 0,
        "delay_per_env_phase": True,
        "history_length": 0,
        "flatten_history_dim": True,
    }
    if target_terms["obstacle_mode"] != expected_mode_term:
        return reject("target obstacle_mode observation differs from the fixed scalar contract")

    source_order = tuple(source_actor_term_order)
    target_order = tuple(target_actor_term_order)
    source_actor = _actor_transfer_topology(source_contract)
    target_actor_abi = _actor_transfer_topology(target_contract)
    if source_actor is None or target_actor_abi is None:
        return reject("source or target task lacks a signed actor ABI")
    source_terms = source_actor["actor_observations"].get("terms", {})
    target_actor_terms = target_actor_abi["actor_observations"].get("terms", {})
    if set(source_order) != set(source_terms) or len(source_order) != len(source_terms):
        return reject("registered locomotion observation order does not match the source contract")
    if set(target_order) != set(target_actor_terms) or len(target_order) != len(target_actor_terms):
        return reject("Generalist observation order does not match the target contract")
    if target_order != (*source_order, "obstacle_mode"):
        return reject("Generalist must append obstacle_mode after the locomotion observations")

    stripped_topology = {
        **dict(target_topology),
        "observations": {
            **dict(observations),
            "actor": {
                **dict(target_actor_group),
                "terms": {
                    name: term for name, term in target_terms.items() if name != "obstacle_mode"
                },
            },
        },
    }
    stripped_contract = {**dict(target_contract), "topology": stripped_topology}
    stripped_match = classify_actor_transfer_compatibility(source_contract, stripped_contract)
    if not stripped_match.compatible:
        return reject("removing only obstacle_mode does not restore the source actor ABI")

    return ActorTransferCompatibility(
        True,
        "locomotion actor ABI preserved with an appended zero obstacle_mode input (mode 0)",
        exact.source_actor_abi_sha256,
        exact.target_actor_abi_sha256,
    )


def classify_task_contract_compatibility(
    checkpoint_contract: Mapping[str, Any] | None,
    current_contract: Mapping[str, Any] | None,
) -> TaskContractCompatibility:
    """Classify exact compatibility, including one audited v1 migration.

    Early v1 checkpoints were produced before registered configurations carried
    ``task_id``.  They are eligible only when replacing that ``null`` value with
    the current requested task makes the complete signed topology identical.
    This is deliberately not a generic compatibility override.
    """
    checkpoint_sha = _contract_sha256(checkpoint_contract)
    current_sha = _contract_sha256(current_contract)
    if not isinstance(checkpoint_contract, Mapping):
        return TaskContractCompatibility(
            TaskContractStatus.LEGACY,
            "checkpoint lacks the task topology contract",
            checkpoint_sha,
            current_sha,
        )
    if not isinstance(current_contract, Mapping):
        return TaskContractCompatibility(
            TaskContractStatus.INCOMPATIBLE,
            "current task topology contract is unavailable",
            checkpoint_sha,
            current_sha,
        )
    if dict(checkpoint_contract) == dict(current_contract):
        return TaskContractCompatibility(
            TaskContractStatus.CURRENT,
            "checkpoint task topology exactly matches the active task",
            checkpoint_sha,
            current_sha,
        )

    checkpoint_topology = checkpoint_contract.get("topology")
    current_topology = current_contract.get("topology")
    if (
        checkpoint_contract.get("schema_version") == TASK_CONTRACT_SCHEMA_VERSION
        and checkpoint_contract.get("id") == TASK_CONTRACT_ID
        and current_contract.get("schema_version") == TASK_CONTRACT_SCHEMA_VERSION
        and current_contract.get("id") == TASK_CONTRACT_ID
        and isinstance(checkpoint_topology, Mapping)
        and isinstance(current_topology, Mapping)
        and checkpoint_topology.get("task_id") is None
        and isinstance(current_topology.get("task_id"), str)
        and current_topology["task_id"]
        and checkpoint_sha == _topology_sha256(checkpoint_topology)
    ):
        normalized_topology = dict(checkpoint_topology)
        normalized_topology["task_id"] = current_topology["task_id"]
        normalized_checkpoint = dict(checkpoint_contract)
        normalized_checkpoint["topology"] = normalized_topology
        normalized_checkpoint["topology_sha256"] = _topology_sha256(normalized_topology)
        if normalized_checkpoint == dict(current_contract):
            return TaskContractCompatibility(
                TaskContractStatus.MIGRATED_V1_NULL_TASK_ID,
                "schema-v1 checkpoint omitted task_id; all other signed topology fields match",
                checkpoint_sha,
                current_sha,
            )

    return TaskContractCompatibility(
        TaskContractStatus.INCOMPATIBLE,
        "checkpoint task topology differs from the active observations, rewards, commands, "
        "events, terminations, or task identity",
        checkpoint_sha,
        current_sha,
    )


def task_contracts_compatible(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> bool:
    """Return whether contracts are exact or use the one audited v1 migration."""
    return classify_task_contract_compatibility(left, right).is_compatible


def require_current_task_contract(
    contract: Mapping[str, Any] | None, cfg: Any
) -> TaskContractCompatibility:
    """Fail before rollout when a checkpoint predates the active task topology."""
    compatibility = classify_task_contract_compatibility(contract, current_task_contract(cfg))
    if compatibility.is_compatible:
        return compatibility
    if compatibility.status == TaskContractStatus.LEGACY:
        raise ValueError(
            "checkpoint lacks the task topology contract; its observations/rewards may predate "
            "the active task and it must be retrained"
        )
    raise ValueError(
        "checkpoint task topology is incompatible with the active observations/rewards; "
        "retrain the policy for this task schema"
    )


__all__ = [
    "TASK_CONTRACT_ID",
    "TASK_CONTRACT_SCHEMA_VERSION",
    "ActorTransferCompatibility",
    "TaskContractCompatibility",
    "TaskContractStatus",
    "classify_actor_transfer_compatibility",
    "classify_task_contract_compatibility",
    "canonical_task_cfg_for_runtime_cfg",
    "current_task_contract",
    "current_task_contract_for_task",
    "require_current_task_contract",
    "task_contracts_compatible",
    "task_topology",
]

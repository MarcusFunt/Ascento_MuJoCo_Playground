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
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .physics import REWARD_SCHEMA_VERSION

TASK_CONTRACT_SCHEMA_VERSION = 1
TASK_CONTRACT_ID = "ascento_task_topology_v1"


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
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
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
        "observations": {
            str(name): _canonical(group)
            for name, group in observations.items()
        },
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

    return current_task_contract(load_env_cfg(task, play=play))


def task_contracts_compatible(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> bool:
    """Task semantics are an exact ABI, not a best-effort compatibility hint."""
    return isinstance(left, Mapping) and isinstance(right, Mapping) and dict(left) == dict(right)


def require_current_task_contract(
    contract: Mapping[str, Any] | None, cfg: Any
) -> None:
    """Fail before rollout when a checkpoint predates the active task topology."""
    if not isinstance(contract, Mapping):
        raise ValueError(
            "checkpoint lacks the task topology contract; its observations/rewards may predate "
            "the active task and it must be retrained"
        )
    if not task_contracts_compatible(contract, current_task_contract(cfg)):
        raise ValueError(
            "checkpoint task topology is incompatible with the active observations/rewards; "
            "retrain the policy for this task schema"
        )


__all__ = [
    "TASK_CONTRACT_ID",
    "TASK_CONTRACT_SCHEMA_VERSION",
    "current_task_contract",
    "current_task_contract_for_task",
    "require_current_task_contract",
    "task_contracts_compatible",
    "task_topology",
]

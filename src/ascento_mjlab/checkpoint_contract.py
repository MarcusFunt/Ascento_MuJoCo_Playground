"""One validation entry point for checkpoint provenance contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .control_contract import require_current_action_contract
from .plant_contract import require_current_plant_contract
from .task_contract import TaskContractCompatibility, require_current_task_contract


def apply_checkpoint_curriculum_contract(
    cfg: Any, infos: Mapping[str, Any] | None, task: str
) -> bool:
    """Apply checkpoint-recorded curriculum fractions to a copied task config.

    mjlab registers task configs at import time. Evaluator environment
    variables therefore arrive too late to change the cached config; patch only
    these whitelisted curriculum parameters on the deep-copied config instead.
    The caller still performs a strict full-topology contract check afterward.
    """
    values = infos if isinstance(infos, Mapping) else {}
    task_contract = values.get("task_contract")
    topology = task_contract.get("topology") if isinstance(task_contract, Mapping) else None
    if not isinstance(topology, Mapping) or topology.get("task_id") != task:
        return False
    events = topology.get("events")
    initializer = events.get("initialize_world_target") if isinstance(events, Mapping) else None
    params = initializer.get("params") if isinstance(initializer, Mapping) else None
    if not isinstance(params, Mapping):
        return False
    precision = params.get("precision_anchor_fraction")
    recovery = params.get("recovery_retarget_anchor_fraction")
    if (
        not isinstance(precision, (int, float))
        or isinstance(precision, bool)
        or not isinstance(recovery, (int, float))
        or isinstance(recovery, bool)
        or not math.isfinite(float(precision))
        or not math.isfinite(float(recovery))
        or not 0.0 < float(precision) < 1.0
        or not 0.0 < float(recovery) < 1.0
        or float(precision) + float(recovery) >= 1.0
    ):
        return False

    for group_name in ("events", "metrics"):
        group = getattr(cfg, group_name, {}) or {}
        for term in group.values():
            term_params = getattr(term, "params", None)
            if not isinstance(term_params, dict):
                continue
            if "precision_anchor_fraction" in term_params:
                term_params["precision_anchor_fraction"] = float(precision)
            if "recovery_retarget_anchor_fraction" in term_params:
                term_params["recovery_retarget_anchor_fraction"] = float(recovery)
            if "gate_like_fraction" in term_params:
                term_params["gate_like_fraction"] = float(recovery)
            if "curriculum_start_gate_like_fraction" in term_params:
                term_params["curriculum_start_gate_like_fraction"] = float(recovery)
            if "curriculum_final_gate_like_fraction" in term_params:
                term_params["curriculum_final_gate_like_fraction"] = float(recovery)
    return True


def require_current_checkpoint_contracts(
    infos: Mapping[str, Any] | None, cfg: Any
) -> TaskContractCompatibility:
    """Reject a checkpoint before it can run against incompatible semantics."""
    values = infos if isinstance(infos, Mapping) else {}
    require_current_plant_contract(values.get("plant_contract"))
    require_current_action_contract(values.get("action_contract"))
    return require_current_task_contract(values.get("task_contract"), cfg)


__all__ = ["apply_checkpoint_curriculum_contract", "require_current_checkpoint_contracts"]

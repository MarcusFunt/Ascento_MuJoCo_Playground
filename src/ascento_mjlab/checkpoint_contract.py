"""One validation entry point for checkpoint provenance contracts."""

from __future__ import annotations

from typing import Any, Mapping

from .control_contract import require_current_action_contract
from .plant_contract import require_current_plant_contract
from .task_contract import require_current_task_contract


def require_current_checkpoint_contracts(infos: Mapping[str, Any] | None, cfg: Any) -> None:
    """Reject a checkpoint before it can run against incompatible semantics."""
    values = infos if isinstance(infos, Mapping) else {}
    require_current_plant_contract(values.get("plant_contract"))
    require_current_action_contract(values.get("action_contract"))
    require_current_task_contract(values.get("task_contract"), cfg)


__all__ = ["require_current_checkpoint_contracts"]

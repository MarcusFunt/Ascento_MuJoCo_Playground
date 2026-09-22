"""Dashboard-facing task catalog."""

from __future__ import annotations

from typing import Any

_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "Ascento-Balance-Flat",
        "label": "Balance",
        "description": "World-target balance with adaptive 20→60→120→300 s horizons.",
        "supports_horizon": True,
    },
    {
        "id": "Ascento-Balance-Quiet-Flat",
        "label": "Quiet balance",
        "description": "Quiet-balance continuation without adaptive horizon promotion.",
        "supports_horizon": False,
    },
    {
        "id": "Ascento-Balance-Recovery-Flat",
        "label": "Balance recovery",
        "description": "Balance recovery with adaptive horizon and reset-difficulty curricula.",
        "supports_horizon": True,
    },
    {
        "id": "Ascento-Locomotion-Flat",
        "label": "Locomotion",
        "description": "Settle, push, recover, move to a nearby world target, then stop.",
        "supports_horizon": False,
    },
    {
        "id": "Ascento-Velocity-Flat",
        "label": "Velocity",
        "description": "Command tracking with adaptive 20→60→120→300 s horizons.",
        "supports_horizon": True,
    },
    {
        "id": "Ascento-Recovery-Flat",
        "label": "Recovery",
        "description": "Recovery-focused training task.",
        "supports_horizon": False,
    },
    {
        "id": "Ascento-Jump-Flat",
        "label": "Jump",
        "description": "Jump and landing task.",
        "supports_horizon": False,
    },
)


def task_catalog() -> list[dict[str, Any]]:
    return [dict(item) for item in _TASKS]


def task_ids() -> set[str]:
    return {str(item["id"]) for item in _TASKS}


def horizon_task_ids() -> set[str]:
    return {str(item["id"]) for item in _TASKS if item["supports_horizon"]}

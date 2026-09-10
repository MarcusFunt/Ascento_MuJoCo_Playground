"""Immutable public contract for the structured policy-to-target controller."""

from __future__ import annotations

from typing import Any, Mapping

ACTION_SCHEMA_ID = "structured_targets_v1"
ACTION_CONTRACT_SCHEMA_VERSION = 1
ACTION_ORDER = (
    "left_hip_position_target",
    "left_knee_position_target",
    "left_wheel_velocity_target",
    "right_hip_position_target",
    "right_knee_position_target",
    "right_wheel_velocity_target",
)
LEG_POSITION_SCALE_RAD = 1.5707963267948966
WHEEL_VELOCITY_SCALE_RAD_S = 8.0
LEG_KP_NM_RAD = 40.0
LEG_KD_NM_S_RAD = 2.0
WHEEL_KP_NM_S_RAD = 4.0
WHEEL_KI_NM_RAD = 4.0
WHEEL_INTEGRAL_LIMIT_RAD = 5.0
CONTROLLER_REQUEST_LIMIT_NM = 65.0
# MuJoCo's wheel joint convention advances this robot along negative base X for
# a positive joint velocity. Invert both channels so policy-positive is forward.
WHEEL_TARGET_SIGNS = (-1.0, -1.0)


def current_action_contract() -> dict[str, Any]:
    """Return the exact controller ABI required by checkpoints and evidence."""
    return {
        "schema_version": ACTION_CONTRACT_SCHEMA_VERSION,
        "id": ACTION_SCHEMA_ID,
        "action_order": list(ACTION_ORDER),
        "leg_position_scale_rad": LEG_POSITION_SCALE_RAD,
        "wheel_velocity_scale_rad_s": WHEEL_VELOCITY_SCALE_RAD_S,
        "wheel_target_signs": list(WHEEL_TARGET_SIGNS),
        "leg_pd": {"kp_nm_rad": LEG_KP_NM_RAD, "kd_nm_s_rad": LEG_KD_NM_S_RAD},
        "wheel_pi": {
            "kp_nm_s_rad": WHEEL_KP_NM_S_RAD,
            "ki_nm_rad": WHEEL_KI_NM_RAD,
            "integral_limit_rad": WHEEL_INTEGRAL_LIMIT_RAD,
        },
        "controller_request_limit_nm": CONTROLLER_REQUEST_LIMIT_NM,
        "controller_version": 1,
    }


def action_contracts_compatible(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> bool:
    """Require an exact structured-control ABI before comparing artifacts."""
    return isinstance(left, Mapping) and isinstance(right, Mapping) and dict(left) == dict(right)


def require_current_action_contract(contract: Mapping[str, Any] | None) -> None:
    """Reject legacy direct-effort checkpoints before they can produce a rollout."""
    if not isinstance(contract, Mapping):
        raise ValueError(
            "checkpoint lacks the structured action contract; direct-effort checkpoints are not "
            "compatible with structured_targets_v1 and must be retrained"
        )
    if not action_contracts_compatible(contract, current_action_contract()):
        raise ValueError(
            "checkpoint action contract is incompatible with structured_targets_v1; retrain the policy "
            "with this controller schema"
        )


__all__ = [
    "ACTION_CONTRACT_SCHEMA_VERSION",
    "ACTION_ORDER",
    "ACTION_SCHEMA_ID",
    "CONTROLLER_REQUEST_LIMIT_NM",
    "LEG_KD_NM_S_RAD",
    "LEG_KP_NM_RAD",
    "LEG_POSITION_SCALE_RAD",
    "WHEEL_INTEGRAL_LIMIT_RAD",
    "WHEEL_KI_NM_RAD",
    "WHEEL_KP_NM_S_RAD",
    "WHEEL_TARGET_SIGNS",
    "WHEEL_VELOCITY_SCALE_RAD_S",
    "action_contracts_compatible",
    "current_action_contract",
    "require_current_action_contract",
]

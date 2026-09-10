"""Single six-channel action term for structured leg and wheel targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .control_contract import LEG_POSITION_SCALE_RAD, WHEEL_TARGET_SIGNS, WHEEL_VELOCITY_SCALE_RAD_S
from .robot_cfg import JOINT_NAMES

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class StructuredTargetActionCfg(ActionTermCfg):
    """Six normalized actions in the immutable public action order."""

    actuator_names: tuple[str, ...] = JOINT_NAMES

    def build(self, env: ManagerBasedRlEnv) -> "StructuredTargetAction":
        return StructuredTargetAction(self, env)


class StructuredTargetAction(ActionTerm):
    def __init__(self, cfg: StructuredTargetActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        ids, names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        if tuple(names) != JOINT_NAMES:
            raise ValueError(f"structured target order must be {JOINT_NAMES}, got {tuple(names)}")
        self._joint_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._leg_slots = torch.tensor((0, 1, 3, 4), dtype=torch.long, device=self.device)
        self._wheel_slots = torch.tensor((2, 5), dtype=torch.long, device=self.device)
        self._raw_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._position_targets = torch.zeros((self.num_envs, 4), device=self.device)
        self._velocity_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self._wheel_target_signs = torch.tensor(
            WHEEL_TARGET_SIGNS, device=self.device, dtype=self._raw_actions.dtype
        )

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected {self.action_dim} structured actions, got {actions.shape[-1]}"
            )
        self._raw_actions.copy_(torch.clamp(actions, -1.0, 1.0))
        nominal = self._entity.data.default_joint_pos[:, self._joint_ids[self._leg_slots]]
        limits = self._entity.data.joint_pos_limits[:, self._joint_ids[self._leg_slots]]
        self._position_targets.copy_(
            torch.clamp(
                nominal + self._raw_actions[:, self._leg_slots] * LEG_POSITION_SCALE_RAD,
                min=limits[..., 0],
                max=limits[..., 1],
            )
        )
        self._velocity_targets.copy_(
            self._raw_actions[:, self._wheel_slots]
            * WHEEL_VELOCITY_SCALE_RAD_S
            * self._wheel_target_signs
        )

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(
            self._position_targets, joint_ids=self._joint_ids[self._leg_slots]
        )
        self._entity.set_joint_velocity_target(
            self._velocity_targets, joint_ids=self._joint_ids[self._wheel_slots]
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = env_ids if env_ids is not None else slice(None)
        self._raw_actions[ids] = 0.0
        self._position_targets[ids] = 0.0
        self._velocity_targets[ids] = 0.0

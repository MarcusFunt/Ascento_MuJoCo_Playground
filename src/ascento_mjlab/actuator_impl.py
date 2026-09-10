"""mjlab-dependent implementation of the Ascento actuator model.

Kept separate from :mod:`ascento_mjlab.actuator` so importing the public
actuator specification does not import mjlab while its entry-point task loader
is still starting.  This avoids the actuator-first task-registration cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import torch
from mjlab.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
    from mjlab.entity import Entity


def torque_speed_limit(
    peak_torque: float,
    no_load_speed: float,
    controller_speed_limit: float,
    velocity: torch.Tensor,
    requested: torch.Tensor,
) -> torch.Tensor:
    """Apply the transient torque-speed envelope and motoring speed guard."""
    speed = torch.abs(velocity)
    envelope = torch.clamp(1.0 - speed / no_load_speed, min=0.0)
    output = torch.clamp(requested, -peak_torque * envelope, peak_torque * envelope)
    over_speed = speed > controller_speed_limit
    motoring = torch.sign(output) * velocity > 0.0
    return torch.where(over_speed & motoring, 0.0, output)


@dataclass(kw_only=True)
class AscentoTargetActuatorCfg(ActuatorCfg):
    """Physics-rate target controller followed by the Ascento motor model."""

    peak_torque: float
    no_load_speed: float
    controller_speed_limit: float
    response_time: float
    controller: Literal["position_pd", "velocity_pi"]
    kp: float
    kd_or_ki: float
    integral_limit: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.peak_torque <= 0.0:
            raise ValueError("peak_torque must be positive")
        if self.no_load_speed <= 0.0:
            raise ValueError("no_load_speed must be positive")
        if self.controller_speed_limit <= 0.0:
            raise ValueError("controller_speed_limit must be positive")
        if self.response_time < 0.0:
            raise ValueError("response_time must be non-negative")
        if self.controller not in {"position_pd", "velocity_pi"}:
            raise ValueError("controller must be position_pd or velocity_pi")
        if self.kp < 0.0 or self.kd_or_ki < 0.0:
            raise ValueError("controller gains must be non-negative")
        if self.controller == "velocity_pi" and self.integral_limit <= 0.0:
            raise ValueError("velocity_pi requires a positive integral_limit")

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> AscentoTargetActuator:
        return AscentoTargetActuator(self, entity, target_ids, target_names)


class AscentoTargetActuator(Actuator[AscentoTargetActuatorCfg]):
    """PD/PI target control with response dynamics and torque-speed limiting."""

    def __init__(
        self,
        cfg: AscentoTargetActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._filtered: torch.Tensor | None = None
        self._physics_dt: float | None = None
        self._integral: torch.Tensor | None = None
        self._controller_requested: torch.Tensor | None = None
        self._output: torch.Tensor | None = None
        self._work_a: torch.Tensor | None = None
        self._work_b: torch.Tensor | None = None
        self._work_c: torch.Tensor | None = None
        self._work_d: torch.Tensor | None = None
        self._mask_a: torch.Tensor | None = None
        self._mask_b: torch.Tensor | None = None
        self._response_alpha: float | None = None

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        for target_name in target_names:
            self._mjs_actuators.append(
                create_motor_actuator(
                    spec,
                    target_name,
                    effort_limit=self.cfg.peak_torque,
                    transmission_type=self.cfg.transmission_type,
                )
            )

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model,
        data,
        device: str,
    ) -> None:
        self._physics_dt = float(mj_model.opt.timestep)
        super().initialize(mj_model, model, data, device)
        self._filtered = torch.zeros(
            (data.nworld, len(self.target_ids)), dtype=torch.float32, device=device
        )
        self._integral = torch.zeros_like(self._filtered)
        self._controller_requested = torch.zeros_like(self._filtered)
        self._output = torch.zeros_like(self._filtered)
        self._work_a = torch.zeros_like(self._filtered)
        self._work_b = torch.zeros_like(self._filtered)
        self._work_c = torch.zeros_like(self._filtered)
        self._work_d = torch.zeros_like(self._filtered)
        self._mask_a = torch.zeros_like(self._filtered, dtype=torch.bool)
        self._mask_b = torch.zeros_like(self._filtered, dtype=torch.bool)
        self._response_alpha = (
            1.0
            if self.cfg.response_time == 0.0
            else 1.0 - math.exp(-self._physics_dt / self.cfg.response_time)
        )

    @property
    def controller_requested_effort(self) -> torch.Tensor:
        if self._controller_requested is None:
            raise RuntimeError("Actuator must be initialized before reading controller effort")
        return self._controller_requested

    def _ensure_workspace(self) -> None:
        """Allocate persistent work buffers once, including for direct unit construction."""
        if self._filtered is None:
            raise RuntimeError("Actuator must be initialized before compute()")
        for name, dtype in (
            ("_controller_requested", None),
            ("_output", None),
            ("_work_a", None),
            ("_work_b", None),
            ("_work_c", None),
            ("_work_d", None),
            ("_mask_a", torch.bool),
            ("_mask_b", torch.bool),
        ):
            if getattr(self, name, None) is None:
                setattr(self, name, torch.zeros_like(self._filtered, dtype=dtype))
        if getattr(self, "_response_alpha", None) is None:
            if self._physics_dt is None:
                raise RuntimeError("Actuator must be initialized before compute()")
            self._response_alpha = (
                1.0
                if self.cfg.response_time == 0.0
                else 1.0 - math.exp(-self._physics_dt / self.cfg.response_time)
            )

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self._filtered is not None
        self._ensure_workspace()
        assert self._controller_requested is not None
        assert self._output is not None
        assert self._work_a is not None
        assert self._work_b is not None
        assert self._work_c is not None
        assert self._work_d is not None
        assert self._mask_a is not None
        assert self._mask_b is not None
        assert self._response_alpha is not None
        requested = self._controller_requested
        if self.cfg.controller == "position_pd":
            requested.copy_(cmd.position_target).sub_(cmd.pos).mul_(self.cfg.kp)
            requested.add_(cmd.vel, alpha=-self.cfg.kd_or_ki)
        else:
            if self._integral is None:
                raise RuntimeError("Actuator must be initialized before compute()")
            if self._physics_dt is None:
                raise RuntimeError("Actuator must be initialized before compute()")
            error = self._work_a.copy_(cmd.velocity_target).sub_(cmd.vel)
            candidate = self._work_b.copy_(self._integral)
            candidate.add_(error, alpha=self._physics_dt).clamp_(
                -self.cfg.integral_limit, self.cfg.integral_limit
            )
            unsaturated = self._work_c.copy_(error).mul_(self.cfg.kp)
            unsaturated.add_(candidate, alpha=self.cfg.kd_or_ki)
            # Conditional integration avoids increasing wind-up while saturated in the error direction.
            torch.abs(unsaturated, out=self._work_d)
            torch.lt(self._work_d, self.cfg.peak_torque, out=self._mask_a)
            torch.sign(error, out=self._work_d)
            torch.sign(unsaturated, out=requested)
            torch.ne(self._work_d, requested, out=self._mask_b)
            self._mask_a.logical_or_(self._mask_b)
            torch.where(self._mask_a, candidate, self._integral, out=self._integral)
            requested.copy_(error).mul_(self.cfg.kp)
            requested.add_(self._integral, alpha=self.cfg.kd_or_ki)
        requested.clamp_(-self.cfg.peak_torque, self.cfg.peak_torque)

        # Keep all temporaries persistent.  Warp captures this method on every
        # physics tick; allocating a fresh CUDA tensor here eventually exhausts
        # the graph's resident allocation pool during long evaluation runs.
        self._filtered.mul_(1.0 - self._response_alpha)
        self._filtered.add_(requested, alpha=self._response_alpha)
        speed = self._work_a.copy_(cmd.vel).abs_()
        effort_limit = self._work_b.copy_(speed).div_(self.cfg.no_load_speed)
        effort_limit.neg_().add_(1.0).clamp_(min=0.0).mul_(self.cfg.peak_torque)
        output = self._output.copy_(self._filtered)
        torch.minimum(output, effort_limit, out=output)
        self._work_c.copy_(effort_limit).neg_()
        torch.maximum(output, self._work_c, out=output)
        torch.gt(speed, self.cfg.controller_speed_limit, out=self._mask_a)
        self._work_d.copy_(output).mul_(cmd.vel)
        torch.gt(self._work_d, 0.0, out=self._mask_b)
        self._mask_a.logical_and_(self._mask_b)
        output.masked_fill_(self._mask_a, 0.0)
        return output

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        assert self._filtered is not None
        self._filtered[env_ids if env_ids is not None else slice(None)] = 0.0
        assert self._integral is not None and self._controller_requested is not None
        self._integral[env_ids if env_ids is not None else slice(None)] = 0.0
        self._controller_requested[env_ids if env_ids is not None else slice(None)] = 0.0
        assert self._output is not None
        self._output[env_ids if env_ids is not None else slice(None)] = 0.0

"""Deterministic, order-independent scenario materialization."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from .schema import CommandPoint, DisturbanceSpec, FamilySpec, ScenarioSpec, SuiteSpec

RESET_KEYS = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
)


def scenario_seed(suite: SuiteSpec, family_id: str, index: int) -> int:
    payload = f"{suite.suite_id}|{suite.root_seed}|{family_id}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _range(config: dict[str, Any], key: str) -> tuple[float, float]:
    reset = config.get("reset", {})
    raw = reset.get(key, [0.0, 0.0])
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"reset.{key} must be [min, max]")
    return float(raw[0]), float(raw[1])


def _uniform_reset(rng: random.Random, config: dict[str, Any]) -> dict[str, float]:
    return {key: rng.uniform(*_range(config, key)) for key in RESET_KEYS}


def _corner_reset(index: int, config: dict[str, Any]) -> dict[str, float]:
    """Enumerate deterministic low/high combinations with yaw strata.

    The bit pattern covers every bounded reset dimension repeatedly. Yaw is
    stratified across 0, pi/2, pi, and 3pi/2 by default because flat-ground
    balance should not depend on world heading.
    """
    reset: dict[str, float] = {}
    bit = 0
    for key in RESET_KEYS:
        low, high = _range(config, key)
        if key == "yaw":
            strata = config.get("yaw_strata", [0.0, math.pi / 2, math.pi, -math.pi / 2])
            reset[key] = float(strata[index % len(strata)])
            continue
        if low == high:
            reset[key] = low
            continue
        reset[key] = high if ((index >> bit) & 1) else low
        bit += 1
    return reset


def _disturbance(index: int, family: FamilySpec, step_dt: float) -> DisturbanceSpec:
    config = family.config.get("disturbance", {})
    start_s = float(config.get("start_s", 5.0))
    duration_s = float(config.get("duration_s", 0.1))
    directions = [str(x) for x in config.get("directions", ["+x", "-x", "+y", "-y"])]
    delta_vs = [float(x) for x in config.get("equivalent_delta_v", [0.15, 0.30, 0.45])]
    force_values = [float(x) for x in config.get("force_n", [])]
    direction = directions[index % len(directions)]
    level_index = (index // len(directions)) % max(len(delta_vs), len(force_values), 1)
    delta_v = delta_vs[level_index % len(delta_vs)] if delta_vs else 0.0
    force_n = force_values[level_index % len(force_values)] if force_values else None
    return DisturbanceSpec(
        start_step=round(start_s / step_dt),
        duration_steps=max(1, round(duration_s / step_dt)),
        direction=direction,
        equivalent_delta_v=delta_v,
        force_n=force_n,
    )


def _commands(family: FamilySpec, step_dt: float) -> tuple[CommandPoint, ...]:
    timeline = family.config.get("commands", [])
    points: list[CommandPoint] = []
    for item in timeline:
        points.append(
            CommandPoint(
                step=round(float(item["time_s"]) / step_dt),
                name=str(item["name"]),
                values=tuple(float(x) for x in item["values"]),
            )
        )
    points.sort(key=lambda item: item.step)
    return tuple(points)


def materialize_suite(suite: SuiteSpec, step_dt: float) -> list[ScenarioSpec]:
    if step_dt <= 0:
        raise ValueError("step_dt must be positive")
    scenarios: list[ScenarioSpec] = []
    for family in suite.families:
        if family.count < 1:
            raise ValueError(f"Family {family.family_id!r} must contain at least one scenario")
        horizon_steps = max(1, round(family.horizon_s / step_dt))
        for index in range(family.count):
            rng = random.Random(scenario_seed(suite, family.family_id, index))
            if family.kind in {"uniform_reset", "disturbance", "endurance", "command_timeline"}:
                reset = _uniform_reset(rng, family.config)
            elif family.kind == "corners":
                reset = _corner_reset(index, family.config)
            else:
                raise ValueError(f"Unsupported scenario family kind: {family.kind}")
            disturbances = (
                (_disturbance(index, family, step_dt),) if family.kind == "disturbance" else ()
            )
            scenarios.append(
                ScenarioSpec(
                    scenario_id=f"{suite.suite_id}/{family.family_id}/{index:06d}",
                    family=family.family_id,
                    task=suite.task,
                    horizon_steps=horizon_steps,
                    reset=reset,
                    disturbances=disturbances,
                    commands=_commands(family, step_dt),
                    tags=(family.kind,),
                )
            )
    return scenarios

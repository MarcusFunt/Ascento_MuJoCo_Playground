"""Versioned schemas for quantitative evaluation suites and results."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class EvaluationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class DisturbanceSpec:
    start_step: int
    duration_steps: int
    direction: str
    equivalent_delta_v: float = 0.0
    force_n: float | None = None
    torque_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CommandPoint:
    step: int
    name: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: str
    task: str
    horizon_steps: int
    reset: dict[str, float]
    disturbances: tuple[DisturbanceSpec, ...] = ()
    commands: tuple[CommandPoint, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    family: str
    metric: str
    statistic: str
    op: str
    threshold: float
    hard: bool = True


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    kind: str
    count: int
    horizon_s: float
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteSpec:
    schema_version: int
    suite_id: str
    task: str
    root_seed: int
    policy_mode: str
    families: tuple[FamilySpec, ...]
    gates: tuple[GateSpec, ...]
    required_capabilities: tuple[str, ...] = ()

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class EpisodeResult:
    scenario_id: str
    family: str
    success: bool
    termination_reason: str
    episode_steps: int
    metrics: dict[str, float]
    events: dict[str, float] = field(default_factory=dict)


def _family_from_raw(raw: dict[str, Any]) -> FamilySpec:
    known = {"id", "kind", "count", "horizon_s"}
    return FamilySpec(
        family_id=str(raw["id"]),
        kind=str(raw["kind"]),
        count=int(raw["count"]),
        horizon_s=float(raw["horizon_s"]),
        config={k: v for k, v in raw.items() if k not in known},
    )


def load_suite(path: str | Path) -> SuiteSpec:
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    gates = tuple(
        GateSpec(
            gate_id=str(item["id"]),
            family=str(item["family"]),
            metric=str(item["metric"]),
            statistic=str(item["statistic"]),
            op=str(item["op"]),
            threshold=float(item["threshold"]),
            hard=bool(item.get("hard", True)),
        )
        for item in raw.get("gates", [])
    )
    families = tuple(_family_from_raw(item) for item in raw.get("families", []))
    suite = SuiteSpec(
        schema_version=int(raw.get("schema_version", 1)),
        suite_id=str(raw["suite_id"]),
        task=str(raw["task"]),
        root_seed=int(raw.get("root_seed", 0)),
        policy_mode=str(raw.get("policy_mode", "deterministic")),
        families=families,
        gates=gates,
        required_capabilities=tuple(str(x) for x in raw.get("required_capabilities", [])),
    )
    if suite.schema_version != 1:
        raise ValueError(f"Unsupported evaluation schema version: {suite.schema_version}")
    if suite.policy_mode not in {"deterministic", "stochastic"}:
        raise ValueError(f"Unsupported policy mode: {suite.policy_mode}")
    if len({family.family_id for family in suite.families}) != len(suite.families):
        raise ValueError("Family IDs must be unique")
    return suite


def canonical_scenario_json(scenario: ScenarioSpec) -> str:
    return json.dumps(asdict(scenario), sort_keys=True, separators=(",", ":"))


def scenarios_sha256(scenarios: list[ScenarioSpec]) -> str:
    digest = hashlib.sha256()
    for scenario in scenarios:
        digest.update(canonical_scenario_json(scenario).encode())
        digest.update(b"\n")
    return digest.hexdigest()

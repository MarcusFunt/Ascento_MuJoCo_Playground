"""Persistent evaluation artifacts: SQLite plus canonical JSON manifests."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import EpisodeResult, ScenarioSpec, SuiteSpec, canonical_scenario_json


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS scenario (
      scenario_id TEXT PRIMARY KEY,
      family TEXT NOT NULL,
      task TEXT NOT NULL,
      horizon_steps INTEGER NOT NULL,
      spec_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS episode (
      scenario_id TEXT PRIMARY KEY REFERENCES scenario(scenario_id),
      success INTEGER NOT NULL,
      termination_reason TEXT NOT NULL,
      episode_steps INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS episode_metric (
      scenario_id TEXT NOT NULL REFERENCES episode(scenario_id),
      name TEXT NOT NULL,
      value REAL,
      PRIMARY KEY (scenario_id, name)
    );
    CREATE TABLE IF NOT EXISTS event (
      scenario_id TEXT NOT NULL REFERENCES episode(scenario_id),
      name TEXT NOT NULL,
      value REAL,
      PRIMARY KEY (scenario_id, name)
    );
    """
    )
    return connection


def write_results_database(
    path: Path, scenarios: list[ScenarioSpec], results: list[EpisodeResult]
) -> None:
    connection = initialize_database(path)
    try:
        with connection:
            for scenario in scenarios:
                connection.execute(
                    "INSERT OR REPLACE INTO scenario VALUES (?, ?, ?, ?, ?)",
                    (
                        scenario.scenario_id,
                        scenario.family,
                        scenario.task,
                        scenario.horizon_steps,
                        canonical_scenario_json(scenario),
                    ),
                )
            for result in results:
                connection.execute(
                    "INSERT OR REPLACE INTO episode VALUES (?, ?, ?, ?)",
                    (
                        result.scenario_id,
                        int(result.success),
                        result.termination_reason,
                        result.episode_steps,
                    ),
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO episode_metric VALUES (?, ?, ?)",
                    [
                        (
                            result.scenario_id,
                            name,
                            value if math.isfinite(value) else None,
                        )
                        for name, value in result.metrics.items()
                    ],
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO event VALUES (?, ?, ?)",
                    [
                        (
                            result.scenario_id,
                            name,
                            value if math.isfinite(value) else None,
                        )
                        for name, value in result.events.items()
                    ],
                )
    finally:
        connection.close()


def write_resolved_scenarios(path: Path, scenarios: list[ScenarioSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(canonical_scenario_json(scenario))
            handle.write("\n")


def write_suite_snapshot(path: Path, suite: SuiteSpec) -> None:
    write_json(path, asdict(suite))

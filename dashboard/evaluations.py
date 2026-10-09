"""Read-only discovery for immutable evaluator artifacts."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

_COMPLETED_STATUSES = {"PASS", "FAIL", "INCOMPLETE", "INVALID"}
_REQUIRED_COMPLETE_FILES = (
    "manifest.json",
    "suite.json",
    "resolved_scenarios.jsonl",
    "summary.json",
    "gate.json",
    "consistency.json",
    "results.sqlite",
)

_GATE_OPERATORS = {
    ">=": lambda observed, threshold: observed >= threshold,
    "<=": lambda observed, threshold: observed <= threshold,
    ">": lambda observed, threshold: observed > threshold,
    "<": lambda observed, threshold: observed < threshold,
    "==": lambda observed, threshold: observed == threshold,
}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)
    if not isinstance(value, dict):
        return None, "JSON artifact must contain an object"
    return _clean(value), None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _gate_integrity(suite: dict[str, Any], gate: dict[str, Any]) -> tuple[str | None, str | None]:
    """Check that every declared gate has one internally consistent result."""
    definitions = suite.get("gates")
    results = gate.get("gates")
    if not isinstance(definitions, list) or not isinstance(results, list):
        return "INCOMPLETE", "suite or gate artifact has no gate list"
    if not definitions:
        return None, None
    definition_ids = [
        item.get("id") or item.get("gate_id")
        for item in definitions
        if isinstance(item, dict)
    ]
    result_ids = [item.get("gate_id") or item.get("id") for item in results if isinstance(item, dict)]
    if len(definition_ids) != len(definitions) or len(result_ids) != len(results):
        return "INVALID", "gate rows must be objects with stable identifiers"
    if any(not isinstance(item, str) or not item for item in (*definition_ids, *result_ids)):
        return "INVALID", "gate identifiers must be non-empty strings"
    if len(set(definition_ids)) != len(definition_ids) or len(set(result_ids)) != len(result_ids):
        return "INVALID", "gate identifiers are duplicated"
    if set(result_ids) != set(definition_ids):
        return "INCOMPLETE", "gate results do not cover every suite gate"
    by_id = {item.get("gate_id") or item.get("id"): item for item in results}
    for definition in definitions:
        gate_id = definition.get("id") or definition.get("gate_id")
        result = by_id[gate_id]
        observed = result.get("observed")
        passed = result.get("passed")
        if observed is None or isinstance(observed, bool) or not isinstance(observed, (int, float)):
            return "INCOMPLETE", f"gate {gate_id} has no finite observation"
        if not math.isfinite(float(observed)) or not isinstance(passed, bool):
            return "INCOMPLETE", f"gate {gate_id} has no finite observation or boolean verdict"
        operator = definition.get("op")
        threshold = definition.get("threshold")
        if operator not in _GATE_OPERATORS or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            return "INVALID", f"suite gate {gate_id} has an unsupported operator or threshold"
        if (
            result.get("op") != operator
            or result.get("threshold") != threshold
            or result.get("hard") is not bool(definition.get("hard", True))
            or not _GATE_OPERATORS[operator](float(observed), float(threshold)) == passed
        ):
            return "INVALID", f"gate result {gate_id} disagrees with the frozen suite definition"
    return None, None


def _evaluation_path(root: Path, evaluation_id: str) -> Path:
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / evaluation_id).resolve()
    if not _inside(candidate, resolved_root):
        raise ValueError("evaluation path is outside the approved evaluation root")
    if not candidate.is_dir():
        raise ValueError("evaluation was not found")
    return candidate


def _summary(directory: Path, root: Path) -> dict[str, Any]:
    manifest, manifest_error = _json(directory / "manifest.json")
    suite, suite_error = _json(directory / "suite.json")
    gate, gate_error = _json(directory / "gate.json")
    consistency, consistency_error = _json(directory / "consistency.json")
    if manifest_error or suite_error or gate_error or consistency_error:
        integrity_error = "; ".join(
            error for error in (manifest_error, suite_error, gate_error, consistency_error) if error
        )
        status = "INVALID"
    else:
        integrity_error = None
        gate_status = str((gate or {}).get("status", "")).upper()
        complete_files = all((directory / name).is_file() for name in _REQUIRED_COMPLETE_FILES)
        gate_integrity_status, gate_integrity_error = (
            _gate_integrity(suite, gate)
            if isinstance(suite, dict) and isinstance(gate, dict)
            else ("INCOMPLETE", "suite or gate artifact is missing")
        )
        if gate_status == "INVALID" or gate_integrity_status == "INVALID" or (consistency is not None and consistency.get("passed") is False):
            status = "INVALID"
            integrity_error = gate_integrity_error if gate_integrity_status == "INVALID" else integrity_error
        elif gate_integrity_status == "INCOMPLETE":
            status = "INCOMPLETE"
            integrity_error = gate_integrity_error
        elif gate_status == "INCOMPLETE":
            status = "INCOMPLETE"
        elif gate_status not in _COMPLETED_STATUSES or not complete_files:
            status = "INCOMPLETE"
        elif not isinstance(suite, dict):
            status = "INCOMPLETE"
        elif not suite.get("gates"):
            status = "DIAGNOSTIC_ONLY"
        else:
            status = gate_status

    gate_rows = (gate or {}).get("gates")
    gate_rows = gate_rows if isinstance(gate_rows, list) else []
    hard_gates = [row for row in gate_rows if isinstance(row, dict) and row.get("hard") is True]
    passed = sum(1 for row in hard_gates if row.get("passed") is True)
    failed = sum(
        1 for row in hard_gates if row.get("passed") is False and row.get("observed") is not None
    )
    unavailable = sum(1 for row in hard_gates if row.get("observed") is None)
    evidence_class = "diagnostic_only" if isinstance(suite, dict) and not suite.get("gates") else "quantitative"
    evaluation_id = directory.resolve().relative_to(root.resolve()).as_posix()
    manifest = manifest or {}
    suite = suite or {}
    return {
        "evaluation_id": evaluation_id,
        "suite_id": manifest.get("suite_id") or suite.get("suite_id"),
        "task": manifest.get("task") or suite.get("task"),
        "checkpoint": manifest.get("checkpoint"),
        "experiment_id": manifest.get("experiment_id"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "suite_sha256": manifest.get("suite_sha256"),
        "resolved_scenarios_sha256": manifest.get("resolved_scenarios_sha256"),
        "repository_commit": manifest.get("repository_commit"),
        "scenario_count": manifest.get("scenario_count"),
        "status": status,
        "validity": "invalid" if status == "INVALID" else "incomplete" if status == "INCOMPLETE" else "valid",
        "evidence_class": evidence_class,
        "hard_gates_passed": passed,
        "hard_gates_total": len(hard_gates),
        "hard_gates_failed": failed,
        "hard_gates_unavailable": unavailable,
        "finished_at_utc": manifest.get("finished_at_utc"),
        "artifact_mtime": directory.stat().st_mtime,
        "integrity_error": integrity_error,
        "path": evaluation_id,
    }


def discover_evaluations(root: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    """Index only known evaluator manifest/suite/gate directories under root."""
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        return []
    candidates: set[Path] = set()
    try:
        for filename in ("manifest.json", "suite.json", "gate.json"):
            candidates.update(path.parent for path in resolved_root.rglob(filename) if path.is_file())
    except OSError:
        return []
    directories = [path for path in candidates if _inside(path, resolved_root)]
    directories.sort(
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for directory in directories[: max(1, min(limit, 5000))]:
        try:
            results.append(_summary(directory, resolved_root))
        except OSError:
            continue
    return results


def evaluation_detail(root: Path, evaluation_id: str) -> dict[str, Any]:
    directory = _evaluation_path(root, evaluation_id)
    summary = _summary(directory, root.expanduser().resolve())
    payload: dict[str, Any] = {"evaluation": summary}
    for key, filename in (
        ("manifest", "manifest.json"),
        ("suite", "suite.json"),
        ("summary", "summary.json"),
        ("gate", "gate.json"),
        ("consistency", "consistency.json"),
    ):
        value, error = _json(directory / filename)
        payload[key] = value
        if error:
            payload.setdefault("artifact_errors", {})[filename] = error
    return payload


def compare_evaluations(root: Path, baseline_id: str, candidate_id: str) -> dict[str, Any]:
    """Compare completed artifacts only when scenario and runtime contracts match."""
    baseline_path = _evaluation_path(root, baseline_id)
    candidate_path = _evaluation_path(root, candidate_id)
    baseline = _summary(baseline_path, root.expanduser().resolve())
    candidate = _summary(candidate_path, root.expanduser().resolve())
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        if summary["status"] not in {"PASS", "FAIL"} or summary["evidence_class"] != "quantitative":
            raise ValueError(f"{label} must be a complete, valid, quantitative evaluation")

    baseline_manifest, baseline_error = _json(baseline_path / "manifest.json")
    candidate_manifest, candidate_error = _json(candidate_path / "manifest.json")
    if baseline_error or candidate_error or not baseline_manifest or not candidate_manifest:
        raise ValueError("comparison requires readable evaluation manifests")
    identity_keys = ("suite_id", "suite_sha256", "resolved_scenarios_sha256", "task")
    if any(
        not baseline_manifest.get(key)
        or baseline_manifest.get(key) != candidate_manifest.get(key)
        for key in identity_keys
    ):
        raise ValueError("evaluations do not share the same suite, resolved scenarios and task")
    if baseline_manifest.get("task_contract") != candidate_manifest.get("task_contract"):
        raise ValueError("evaluations use different observation or task contracts")
    for label, manifest in (("baseline", baseline_manifest), ("candidate", candidate_manifest)):
        compatibility = manifest.get("checkpoint_task_compatibility_detail")
        if not isinstance(compatibility, dict) or compatibility.get("compatible") is not True:
            raise ValueError(f"{label} checkpoint task compatibility is missing or incompatible")

    from ascento_mjlab.evaluation.compare import compare

    result = compare(baseline_path, candidate_path)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": result,
        "read_only": True,
        "promotion_eligible": False,
        "promotion_note": "This paired comparison does not authorize or start held-out evaluation.",
    }


def evaluation_scenarios(
    root: Path,
    evaluation_id: str,
    *,
    metric: str | None = None,
    direction: str = "high",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    directory = _evaluation_path(root, evaluation_id)
    if direction not in {"high", "low"}:
        raise ValueError("direction must be 'high' or 'low'")
    database = directory / "results.sqlite"
    if not database.is_file():
        raise ValueError("scenario results are incomplete")
    safe_limit = max(1, min(int(limit), 500))
    safe_offset = max(0, int(offset))
    metric_order = "DESC" if direction == "high" else "ASC"
    connection = None
    try:
        connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
        total = int(connection.execute("SELECT COUNT(*) FROM episode").fetchone()[0])
        rows = connection.execute(
            f"""SELECT s.scenario_id, s.family, s.spec_json,
                           e.success, e.termination_reason, e.episode_steps, m.value
                    FROM episode AS e
                    JOIN scenario AS s ON s.scenario_id=e.scenario_id
                    LEFT JOIN episode_metric AS m
                      ON m.scenario_id=e.scenario_id AND (? IS NOT NULL AND m.name=?)
                    ORDER BY e.success ASC, m.value {metric_order}, s.scenario_id ASC
                    LIMIT ? OFFSET ?""",
            (metric, metric, safe_limit, safe_offset),
        ).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"scenario results are unreadable: {error}") from error
    finally:
        if connection is not None:
            connection.close()
    scenarios = []
    for scenario_id, family, spec_json, success, reason, steps, metric_value in rows:
        try:
            spec = json.loads(spec_json)
        except (TypeError, json.JSONDecodeError):
            spec = None
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "family": family,
                "success": bool(success),
                "termination_reason": reason,
                "episode_steps": steps,
                "metric": metric,
                "metric_value": _clean(metric_value),
                "spec": _clean(spec),
            }
        )
    return {"evaluation_id": evaluation_id, "metric": metric, "direction": direction, "total": total, "limit": safe_limit, "offset": safe_offset, "scenarios": scenarios}


def discover_suites(suite_root: Path, evaluation_root: Path) -> list[dict[str, Any]]:
    """List packaged suite definitions and immutable snapshots found in results."""
    entries: dict[str, dict[str, Any]] = {}
    if suite_root.is_dir():
        try:
            for path in sorted(suite_root.glob("*.toml")):
                if not _inside(path, suite_root):
                    continue
                try:
                    import tomllib

                    with path.open("rb") as handle:
                        raw = tomllib.load(handle)
                    suite_id = raw.get("suite_id")
                    if not isinstance(suite_id, str):
                        continue
                    entries[suite_id] = {
                        "suite_id": suite_id,
                        "task": raw.get("task"),
                        "schema_version": raw.get("schema_version", 1),
                        "family_count": len(raw.get("families", [])),
                        "scenario_count": sum(
                            int(family.get("count", 0)) for family in raw.get("families", [])
                        ),
                        "gate_count": len(raw.get("gates", [])),
                        "evidence_class": "diagnostic_only" if not raw.get("gates") else "quantitative",
                        "source": "packaged_definition",
                        "path": path.name,
                    }
                except (OSError, ValueError, TypeError):
                    continue
        except OSError:
            pass
    for item in discover_evaluations(evaluation_root):
        suite_id = item.get("suite_id")
        if not isinstance(suite_id, str) or suite_id in entries:
            continue
        detail = evaluation_detail(evaluation_root, str(item["evaluation_id"]))
        suite = detail.get("suite") or {}
        entries[suite_id] = {
            "suite_id": suite_id,
            "task": item.get("task"),
            "schema_version": suite.get("schema_version"),
            "family_count": len(suite.get("families", [])) if isinstance(suite.get("families"), list) else None,
            "scenario_count": item.get("scenario_count"),
            "gate_count": len(suite.get("gates", [])) if isinstance(suite.get("gates"), list) else None,
            "evidence_class": item.get("evidence_class"),
            "source": "evaluation_snapshot",
            "path": None,
        }
    return sorted(entries.values(), key=lambda item: (str(item.get("task") or ""), str(item["suite_id"])))

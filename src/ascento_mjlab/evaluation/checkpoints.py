"""Screen and rank multiple checkpoints on a common immutable suite."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .cli import _device, evaluate, resolve_suite_path
from .schema import EvaluationStatus

STATUS_RANK = {
    EvaluationStatus.PASS.value: 3,
    EvaluationStatus.FAIL.value: 2,
    EvaluationStatus.INCOMPLETE.value: 1,
    EvaluationStatus.INVALID.value: 0,
}


def _expand(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.update(Path(match) for match in matches)
        else:
            candidate = Path(pattern)
            if candidate.is_file():
                paths.add(candidate)
    return sorted(paths)


def _score(eval_dir: Path) -> tuple[float, float, float]:
    gate = json.loads((eval_dir / "gate.json").read_text())
    summary = json.loads((eval_dir / "summary.json").read_text())
    status_score = float(STATUS_RANK.get(gate.get("status", "INVALID"), 0))
    success_rates = []
    tilt = []
    for family in summary.values():
        if "success" in family and family["success"].get("success_rate") is not None:
            success_rates.append(float(family["success"]["success_rate"]))
        if "max_tilt" in family and family["max_tilt"].get("p95") is not None:
            tilt.append(float(family["max_tilt"]["p95"]))
    success = sum(success_rates) / len(success_rates) if success_rates else -1.0
    quality = -(sum(tilt) / len(tilt)) if tilt else -1.0e9
    return status_score, success, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths or glob patterns")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("evaluations/checkpoint_screen"))
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    checkpoints = _expand(args.checkpoints)
    if not checkpoints:
        parser.error("no checkpoints matched")
    suite_path = resolve_suite_path(args.suite)
    ranked = []
    for checkpoint in checkpoints:
        status, output = evaluate(
            checkpoint=checkpoint,
            suite_path=suite_path,
            output_base=args.output_root,
            batch_size=args.batch_size,
            device=_device(args.device),
        )
        ranked.append((checkpoint, status, output, _score(output)))

    ranked.sort(key=lambda row: row[3], reverse=True)
    print("\nCheckpoint ranking")
    print("==================")
    for index, (checkpoint, status, output, score) in enumerate(ranked[: args.top], start=1):
        print(f"{index:2d}. {checkpoint.name:24s} {status.value:10s} score={score}  {output}")


if __name__ == "__main__":
    main()

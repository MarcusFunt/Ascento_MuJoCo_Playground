"""Run a bounded, sequential waypoint-heading mixture sweep on the WSL GPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "ascento-gpu"
PARENT_CHECKPOINT = ROOT / "logs/rsl_rl/20261006_195013_waypoint-hold-ablation-treatment-300i-seed73_6f8890c9/ascento_locomotion_flat/2026-10-06_19-50-24/model_299.pt"
SWEEP_ROOT = ROOT / "checkpoints/overnight_waypoint_heading_sweep"
EVALUATION_ROOT = ROOT / "evaluations"
SUMMARY_PATH = SWEEP_ROOT / "summary.json"
LOG_PATH = SWEEP_ROOT / "orchestrator.log"
WALL_BUDGET_SECONDS = 9 * 60 * 60
TRAINING_ITERATIONS = 5_000
EXTENDED_ITERATIONS = 8_000
ENVIRONMENTS = 512
SEED = 73
POLL_SECONDS = 45

ARMS = (
    ("control_0pct", "Ascento-Locomotion-Gate-Hold-Control-Flat", 0.0),
    ("turn_12_5pct", "Ascento-Locomotion-Gate-Hold-Turn-12-Flat", 0.125),
    ("turn_25pct", "Ascento-Locomotion-Gate-Hold-Turn-25-Flat", 0.25),
)
SUITES = (
    ("locomotion_sequence_gate_v1", "Ascento-Locomotion-Flat"),
    ("locomotion_waypoint_gate_v2", "Ascento-Locomotion-Flat"),
    ("balance_quiet_quality_regression_v1", "Ascento-Balance-Quiet-Flat"),
    ("balance_gate_v5", "Ascento-Balance-Flat"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


class Sweep:
    def __init__(self) -> None:
        if os.environ.get("ASCENTO_GPU_BOOTSTRAPPED") != "1":
            raise RuntimeError(
                "launch this sweep with scripts/ascento-gpu python scripts/overnight_waypoint_heading_sweep.py"
            )
        self.started = time.monotonic()
        self.deadline = self.started + WALL_BUDGET_SECONDS
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
        self.env["ASCENTO_ARTIFACT_ROOT"] = str(ROOT / "logs/rsl_rl")
        self.env["ASCENTO_EVALUATION_ROOT"] = str(EVALUATION_ROOT)
        self.env["PYTHONUNBUFFERED"] = "1"
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "started_at_utc": utc_now(),
            "deadline_hours": WALL_BUDGET_SECONDS / 3600,
            "source_checkpoint": str(PARENT_CHECKPOINT),
            "seed": SEED,
            "environments": ENVIRONMENTS,
            "initial_iterations": TRAINING_ITERATIONS,
            "extended_iterations": EXTENDED_ITERATIONS,
            "arms": {},
            "comparisons": [],
            "extension_decision": None,
            "state": "running",
        }
        atomic_json(SUMMARY_PATH, self.data)

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def save(self) -> None:
        self.data["updated_at_utc"] = utc_now()
        self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 1)
        self.data["remaining_seconds"] = max(0, round(self.remaining(), 1))
        atomic_json(SUMMARY_PATH, self.data)

    def command(
        self,
        args: list[str],
        *,
        allowed_codes: tuple[int, ...] = (0,),
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if self.remaining() <= 60:
            raise TimeoutError("nine-hour sweep deadline reached")
        command = [str(CLI), *args]
        log("COMMAND " + " ".join(command))
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=min(timeout or self.remaining() - 60, self.remaining() - 60),
            check=False,
        )
        if completed.stdout.strip():
            log("STDOUT " + completed.stdout.strip()[-6000:])
        if completed.stderr.strip():
            log("STDERR " + completed.stderr.strip()[-6000:])
        if completed.returncode not in allowed_codes:
            raise RuntimeError(
                f"command exited {completed.returncode}: {' '.join(command)}"
            )
        return completed

    def transfer(self, source: Path, task: str, output: Path) -> Path:
        if output.exists():
            return output
        self.command(
            [
                "tools",
                "initialize-transfer",
                "--",
                "--source",
                str(source),
                "--task",
                task,
                "--output",
                str(output),
                "--device",
                "cuda:0",
            ]
        )
        if not output.is_file():
            raise RuntimeError(f"actor transfer did not create {output}")
        return output

    def start_training(
        self,
        *,
        arm_name: str,
        task: str,
        fraction: float,
        parent_checkpoint: Path,
        iterations: int,
        parent_run_id: str | None = None,
    ) -> tuple[str, Path]:
        display_name = f"waypoint-heading {arm_name} {iterations}i seed{SEED}"
        resume_checkpoint_name = parent_checkpoint.name
        args = [
            "run",
            "start",
            "--task",
            task,
            "--display-name",
            display_name,
            "--purpose",
            "ablation",
            "--tag",
            "waypoint",
            "--tag",
            "quarter-turn-heading-sweep",
            "--notes",
            f"Only quarter_turn_heading_fraction={fraction} differs across sweep arms; seed={SEED}, envs={ENVIRONMENTS}.",
            "--parent-checkpoint",
            str(parent_checkpoint),
        ]
        if parent_run_id:
            args.extend(["--parent-run-id", parent_run_id])
        args.extend(
            [
                "--envs",
                str(ENVIRONMENTS),
                "--iterations",
                str(iterations),
                "--seed",
                str(SEED),
                "--json",
                "--",
                "--agent.resume",
                "True",
                "--agent.load-run",
                "_resume_parent",
                "--agent.load-checkpoint",
                resume_checkpoint_name,
                "--agent.save-interval",
                "500",
            ]
        )
        created = json.loads(self.command(args).stdout)
        run_id = str(created["id"])
        status_path = self.wait_for_run(run_id)
        status = json.loads(status_path.read_text())
        if status.get("exit_code") != 0 or status.get("state") != "finished":
            raise RuntimeError(
                f"training run {run_id} ended with state={status.get('state')} exit={status.get('exit_code')}"
            )
        relative_checkpoint = status.get("checkpoint_path")
        if not relative_checkpoint:
            raise RuntimeError(f"run {run_id} did not record a final checkpoint")
        run_directory = Path(status["run_directory"])
        checkpoint = run_directory / relative_checkpoint
        if not checkpoint.is_file():
            candidates = sorted(
                (run_directory / "ascento_locomotion_flat").glob("*/model_*.pt"),
                key=lambda path: int(re.search(r"model_(\d+)\.pt$", path.name).group(1)),
            )
            if not candidates:
                raise RuntimeError(f"run {run_id} has no saved model checkpoint")
            checkpoint = candidates[-1]
        log(f"TRAINING_COMPLETE run_id={run_id} checkpoint={checkpoint}")
        self.data.setdefault("run_ids", []).append(run_id)
        self.save()
        return run_id, checkpoint

    def wait_for_run(self, run_id: str) -> Path:
        stop_sent = False
        while self.remaining() > 60:
            matches = list((ROOT / "logs/rsl_rl").glob(f"*{run_id}*/run_status.json"))
            if matches:
                status_path = matches[0]
                try:
                    status = json.loads(status_path.read_text())
                except (OSError, json.JSONDecodeError):
                    status = {}
                if status.get("state") not in {"starting", "running", "stopping", None}:
                    return status_path
            if self.remaining() < 300 and not stop_sent:
                log(f"DEADLINE_STOP run_id={run_id}")
                self.command(
                    ["run", "stop", run_id, "--reason", "overnight_sweep_deadline"],
                    allowed_codes=(0, 1),
                )
                stop_sent = True
            time.sleep(min(POLL_SECONDS, max(1, self.remaining() - 60)))
        raise TimeoutError(f"deadline reached while run {run_id} was active")

    def evaluate(self, arm_name: str, checkpoint: Path) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for suite, task in SUITES:
            if self.remaining() < 3_000:
                log(f"SKIP_REMAINING_SUITE arm={arm_name} suite={suite} insufficient_deadline")
                break
            transfer_path = (
                SWEEP_ROOT
                / arm_name
                / "eval_transfers"
                / task.lower().replace("ascento-", "").replace("-flat", "").replace("-", "_")
                / "model_000000.pt"
            )
            transfer = self.transfer(checkpoint, task, transfer_path)
            self.command(
                [
                    "evaluate",
                    "run",
                    "--checkpoint",
                    str(transfer),
                    "--suite",
                    suite,
                    "--device",
                    "cuda:0",
                    "--json",
                ],
                allowed_codes=(0, 2),
            )
            evaluation = self.find_evaluation(transfer, suite)
            if evaluation is None:
                raise RuntimeError(f"no completed evaluation manifest for {arm_name} / {suite}")
            result[suite] = evaluation
            self.data["arms"][arm_name]["evaluations"] = result
            self.save()
        return result

    def find_evaluation(self, checkpoint: Path, suite: str) -> dict[str, Any] | None:
        matching: list[Path] = []
        for manifest_path in EVALUATION_ROOT.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("suite_id") != suite:
                continue
            recorded_checkpoint = manifest.get("checkpoint")
            if isinstance(recorded_checkpoint, dict):
                recorded_checkpoint = recorded_checkpoint.get("path")
            if recorded_checkpoint and Path(recorded_checkpoint).resolve() == checkpoint.resolve():
                matching.append(manifest_path)
        if not matching:
            return None
        manifest_path = max(matching, key=lambda path: path.stat().st_mtime)
        manifest = json.loads(manifest_path.read_text())
        gate_path = manifest_path.parent / "gate.json"
        if not gate_path.is_file():
            return None
        gate = json.loads(gate_path.read_text())
        gates = gate.get("gates", [])
        family_counts: dict[str, dict[str, int]] = {}
        for item in gates:
            family = str(item.get("family", "unknown"))
            counts = family_counts.setdefault(family, {"passed": 0, "total": 0, "hard_failed": 0})
            counts["total"] += 1
            if item.get("passed"):
                counts["passed"] += 1
            elif item.get("hard"):
                counts["hard_failed"] += 1
        return {
            "id": manifest_path.parent.name,
            "path": str(manifest_path.parent),
            "suite_id": suite,
            "status": gate.get("status", "INCOMPLETE"),
            "passed_gates": sum(bool(item.get("passed")) for item in gates),
            "total_gates": len(gates),
            "hard_failed_gates": sum(
                bool(item.get("hard")) and not bool(item.get("passed")) for item in gates
            ),
            "family_counts": family_counts,
            "scenario_hash": manifest.get("resolved_scenarios_sha256"),
        }

    def compare(self, baseline: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
        if baseline.get("status") not in {"PASS", "FAIL"} or candidate.get("status") not in {"PASS", "FAIL"}:
            return
        output = EVALUATION_ROOT / "waypoint_heading_sweep" / f"{label}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        self.command(
            ["evaluate", "compare", baseline["id"], candidate["id"], "--output", str(output)],
            allowed_codes=(0, 2),
        )
        self.data["comparisons"].append(str(output))
        self.save()

    def all_four_valid(self, arm: dict[str, Any]) -> bool:
        evaluations = arm.get("evaluations", {})
        return all(
            suite in evaluations and evaluations[suite].get("status") in {"PASS", "FAIL"}
            for suite, _ in SUITES
        )

    def hard_failure_count(self, arm: dict[str, Any], suite: str) -> int:
        return int(arm["evaluations"][suite]["hard_failed_gates"])

    def select_extension(self) -> str | None:
        control = self.data["arms"].get("control_0pct")
        if not control or control.get("nonfinite_seen") or not self.all_four_valid(control):
            return None
        candidates: list[tuple[int, str]] = []
        waypoint_suite = "locomotion_waypoint_gate_v2"
        for name in ("turn_12_5pct", "turn_25pct"):
            arm = self.data["arms"].get(name)
            if not arm or not self.all_four_valid(arm):
                continue
            if arm.get("nonfinite_seen"):
                continue
            if arm["evaluations"][waypoint_suite]["passed_gates"] <= control["evaluations"][waypoint_suite]["passed_gates"]:
                continue
            safety_ok = all(
                self.hard_failure_count(arm, suite) <= self.hard_failure_count(control, suite)
                for suite, _ in SUITES
            )
            if safety_ok:
                candidates.append((arm["evaluations"][waypoint_suite]["passed_gates"], name))
        return max(candidates)[1] if candidates else None

    def run(self) -> None:
        if not CLI.is_file():
            raise FileNotFoundError(f"CUDA-capable project CLI not found: {CLI}")
        if not PARENT_CHECKPOINT.is_file():
            raise FileNotFoundError(f"preserved source checkpoint not found: {PARENT_CHECKPOINT}")
        if (ROOT / ".git/index.lock").exists():
            raise RuntimeError("worktree Git index is locked")
        log(f"SWEEP_START root={ROOT} deadline_hours=9 parent={PARENT_CHECKPOINT}")

        for arm_name, task, fraction in ARMS:
            training_input = (
                SWEEP_ROOT
                / arm_name
                / "ascento_locomotion_flat"
                / "actor_transfer"
                / "model_000000.pt"
            )
            self.transfer(PARENT_CHECKPOINT, task, training_input)
            run_id, checkpoint = self.start_training(
                arm_name=arm_name,
                task=task,
                fraction=fraction,
                parent_checkpoint=training_input,
                iterations=TRAINING_ITERATIONS,
            )
            arm = {
                "task": task,
                "fraction": fraction,
                "run_id": run_id,
                "iterations": TRAINING_ITERATIONS,
                "checkpoint": str(checkpoint),
                "nonfinite_seen": self.nonfinite_seen(checkpoint),
                "evaluations": {},
            }
            self.data["arms"][arm_name] = arm
            self.save()
            self.evaluate(arm_name, checkpoint)

            if arm_name == "control_0pct":
                parent = self.find_parent_evaluation_source()
                for suite, _ in SUITES:
                    if suite in arm["evaluations"] and suite in parent:
                        self.compare(parent[suite], arm["evaluations"][suite], f"control_vs_model299_{suite}")
            else:
                control_evaluations = self.data["arms"].get("control_0pct", {}).get("evaluations", {})
                for suite, _ in SUITES:
                    if suite in arm["evaluations"] and suite in control_evaluations:
                        self.compare(control_evaluations[suite], arm["evaluations"][suite], f"{arm_name}_vs_control_{suite}")

        selected = self.select_extension()
        self.data["extension_decision"] = {
            "selected_arm": selected,
            "rule": "Extend the best nonzero mixture only when all four reports are complete, waypoint gate count exceeds control, and hard-gate failures do not exceed control in any suite.",
            "made_at_utc": utc_now(),
        }
        self.save()

        if selected and self.remaining() > 7_200:
            arm = self.data["arms"][selected]
            _, extended_checkpoint = self.start_training(
                arm_name=f"{selected}_extended",
                task=arm["task"],
                fraction=float(arm["fraction"]),
                parent_checkpoint=Path(arm["checkpoint"]),
                iterations=EXTENDED_ITERATIONS,
                parent_run_id=arm["run_id"],
            )
            self.data["arms"][selected]["extended_checkpoint"] = str(extended_checkpoint)
            self.data["arms"][selected]["extended_iterations"] = EXTENDED_ITERATIONS
            self.data["arms"][selected]["extended_evaluations"] = {}
            self.save()
            # Store the extended arm under its own label so its reports cannot
            # overwrite the fixed 5,000-iteration checkpoint's evidence.
            extended_name = f"{selected}_extended"
            self.data["arms"][extended_name] = {
                **arm,
                "run_id": self.data["run_ids"][-1],
                "iterations": EXTENDED_ITERATIONS,
                "checkpoint": str(extended_checkpoint),
                "evaluations": {},
            }
            self.evaluate(extended_name, extended_checkpoint)
            control = self.data["arms"].get("control_0pct", {})
            for suite, _ in SUITES:
                previous = arm.get("evaluations", {}).get(suite)
                current = self.data["arms"][extended_name].get("evaluations", {}).get(suite)
                if previous and current:
                    self.compare(previous, current, f"{extended_name}_vs_5000i_{suite}")
                control_result = control.get("evaluations", {}).get(suite)
                if control_result and current:
                    self.compare(control_result, current, f"{extended_name}_vs_control_{suite}")
        else:
            self.data["extension_decision"]["reason"] = (
                "No nonzero arm met the improvement and safety rule, or insufficient time remained for a full extension plus evaluation."
            )

        self.data["state"] = "complete"
        self.data["finished_at_utc"] = utc_now()
        self.save()
        log(f"SWEEP_COMPLETE summary={SUMMARY_PATH}")

    def nonfinite_seen(self, checkpoint: Path) -> bool:
        run_status_files = list((ROOT / "logs/rsl_rl").glob("*/run_status.json"))
        for status_file in run_status_files:
            try:
                status = json.loads(status_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if status.get("checkpoint_path") and str(checkpoint).endswith(status["checkpoint_path"]):
                log_path = Path(status["run_directory"]) / "training.log"
                if not log_path.is_file():
                    return False
                values = re.findall(r"Episode_Termination/nonfinite:\s*([0-9.]+)", log_path.read_text(errors="replace"))
                return any(float(value) > 0.0 for value in values)
        return False

    def find_parent_evaluation_source(self) -> dict[str, dict[str, Any]]:
        source_by_suite: dict[str, str] = {
            "locomotion_sequence_gate_v1": "20261006T175934Z_locomotion_sequence_gate_v1_model_299",
            "locomotion_waypoint_gate_v2": "20261006T181729Z_locomotion_waypoint_gate_v2_model_299",
            "balance_quiet_quality_regression_v1": "20261006T184007Z_balance_quiet_quality_regression_v1_model_000299",
            "balance_gate_v5": "20261006T191027Z_balance_gate_v5_model_000299",
        }
        result: dict[str, dict[str, Any]] = {}
        for suite, evaluation_id in source_by_suite.items():
            path = EVALUATION_ROOT / evaluation_id / "manifest.json"
            gate_path = path.parent / "gate.json"
            if not path.is_file() or not gate_path.is_file():
                continue
            manifest = json.loads(path.read_text())
            gate = json.loads(gate_path.read_text())
            gates = gate.get("gates", [])
            result[suite] = {
                "id": evaluation_id,
                "path": str(path.parent),
                "suite_id": suite,
                "status": gate.get("status", "INCOMPLETE"),
                "passed_gates": sum(bool(item.get("passed")) for item in gates),
                "total_gates": len(gates),
                "hard_failed_gates": sum(
                    bool(item.get("hard")) and not bool(item.get("passed")) for item in gates
                ),
                "scenario_hash": manifest.get("resolved_scenarios_sha256"),
            }
        return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the sequential nine-hour waypoint-heading ablation sweep."
    )
    parser.parse_args()
    sweep = Sweep()
    try:
        sweep.run()
    except Exception as error:
        sweep.data["state"] = "failed"
        sweep.data["error"] = f"{type(error).__name__}: {error}"
        sweep.data["finished_at_utc"] = utc_now()
        sweep.save()
        log(f"SWEEP_FAILED {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

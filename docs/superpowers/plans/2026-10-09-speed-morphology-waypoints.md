# Speed Commands, Morphology Telemetry, and Waypoint Goals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate runtime speed-command locomotion, evaluator morphology telemetry, and waypoint-goal viewer/evaluation support onto current `origin/main`.

**Architecture:** Reapply the preserved speed feature as a delta on current main, retaining newer evaluator and dashboard contracts. Add focused morphology metric coverage around the current evaluator and cherry-pick only the three waypoint-goals feature commits, resolving overlaps against current code.

**Tech Stack:** Python, PyTorch, mjlab, FastAPI, React/TypeScript, TOML evaluation suites, pytest, Ruff, TypeScript compiler.

**Spec:** Preserved speed-command stash `recovery-save-runtime-speed-command-20261008`; preserved morphology stash `recovery-save-guard-morphology-20261008`; waypoint design at `docs/superpowers/specs/2026-10-06-waypoint-heading-sweep.md` on `codex/waypoint-goals`.

## Global Constraints

- Keep current main's task, plant, and action contracts authoritative.
- Runtime speed commands must respect the configured speed cap and slew limit.
- Evaluation and viewer controls must use the same task command semantics as training.
- Preserve current dashboard, viewer, Blender, and generalist-locomotion behavior.
- Keep generated training outputs and ignored run artifacts out of source changes.

## Review Focus

- Negative, over-cap, and non-finite manual speed requests - test command validation/clamping behavior.
- Checkpoint and evaluator speed caps differing from process environment - test contract and cap restoration.
- Uneven episode termination and missing contact samples - test telemetry masks and finite output.
- Waypoint index/heading controls at sequence boundaries - test IPC and viewer control validation.
- Stale task registration or task-catalog omissions - test registry/CLI/dashboard agreement.

---

### Task 1: Runtime speed-command locomotion

**Files:** preserved additions/modifications in `src/ascento_mjlab/mdp/commands.py`, observations, rewards, task config/registration, CLI/evaluation, dashboard task/run/viewer controls; suite `benchmarks/suites/locomotion_speed_command_gate_v1.toml`; focused tests under `tests_mjlab/` and `tests_dashboard/`.

**Interfaces:** Produces task `Ascento-Locomotion-Speed-Flat`, command term `AscentoTargetSpeedCommand`, evaluator command `speed_fraction`, and suite `locomotion_speed_command_gate_v1`.

- [x] Run the preserved speed-command unit and evaluator tests against current main and record initial failures.
- [x] Reapply and reconcile the preserved speed-command implementation; add regression coverage for invalid manual speed requests and cached task-contract caps.
- [x] Run focused speed, task-config, CLI, and dashboard tests; run Ruff for changed Python files.

### Task 2: Evaluator morphology telemetry

**Files:** `src/ascento_mjlab/evaluation/runner.py`, `tests_mjlab/evaluation/`.

**Interfaces:** Per-scenario morphology metrics are emitted in evaluator arrays alongside reward-term returns and binary wheel-contact telemetry.

- [x] Add a focused regression test for the current metric schema and edge handling, confirm it fails before the telemetry is complete.
- [x] Reconcile the preserved telemetry delta with speed evaluator changes and current main; ensure all metrics use active-sample masks and stable output keys.
- [x] Run focused evaluator tests and Ruff.

### Task 3: Waypoint goals and heading sweep

**Files:** viewer waypoint IPC/worker, locomotion task setup, evaluation runner/report, dashboard task catalog/API/UI, versioned waypoint suite and sweep script, docs, and focused tests.

**Interfaces:** Viewer clients can select waypoint goals/headings; task and gate `Ascento-Locomotion-Gate-Hold-Flat` is registered; heading sweep is runnable through its versioned suite/script.

- [x] Cherry-pick `3315b34`, `5b5fc3c`, and `08f10c8` individually and resolve conflicts against current main.
- [x] Add/fix focused tests for waypoint control bounds, task registration, suite materialization, and dashboard exposure.
- [x] Run focused viewer/evaluation/dashboard tests, frontend typecheck/build, Ruff, and the full available test suite.

- [x] Review the combined diff for regressions and commit the finished implementation.

# Generalist Driver Evaluation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a reproducible 256-scenario screen that establishes whether the generalist policy can drive 2-3 m routes, turn, complete repeated waypoints, recover from a push, and stop accurately.

**Architecture:** Preserve the historical v1 suite and add a new versioned generalist driver suite using the evaluator's existing waypoint dwell and recovery metrics. Add a scenario-contract test, then screen the exact model 100 and protected actor-only transfer with the same deterministic suite and compare the paired reports.

**Tech Stack:** Python, pytest, TOML benchmark suites, Ascento evaluation CLI, MuJoCo-Warp on CUDA.

**Spec:** User request in this chat: fixed 2-3 m driving, turns, repeated waypoints, recovery, and screening model 100 against the protected actor-only checkpoint.

## Global Constraints

- Keep the legacy `roadrunner_generalist_sequence_gate_v1` definition unchanged so existing reports remain interpretable.
- Use exactly 256 deterministic scenarios and the generalist locomotion task.
- Include 2 m and 3 m target routes, turn commands, a three-waypoint route, and a disturbance recovery family.
- Gate final target error at 5 cm and include explicit raw action clipping and actuator saturation gates.
- Evaluate the exact checkpoint files by SHA-256 in the canonical WSL checkout; compare both on the same resolved scenario hash.

## Review Focus

- First waypoint must be applied at time zero so the reset's short random target cannot make a long-goal scenario pass; test command step zero.
- Each route leg must be 2 m or 3 m; test waypoint coordinates and segment lengths.
- The repeated route must count every waypoint as completed, not only the final target crossing; gate `waypoint_sequence_complete`.
- Recovery evidence must be restricted to disturbed route scenarios; test disturbance presence and gate `recovered`.
- The practical 5 cm limit and saturation controls must be hard gates; assert their metrics and thresholds.

---

### Task 1: Add the immutable generalist driver suite

**Files:**
- Create: `benchmarks/suites/roadrunner_generalist_driver_gate_v1.toml`
- Modify: `tests_mjlab/evaluation/test_scenarios.py`

**Interfaces:**
- Consumes: `load_suite`, `materialize_suite`, `ScenarioSpec.commands`, and existing waypoint/recovery metrics.
- Produces: Suite ID `roadrunner_generalist_driver_gate_v1`, with 4 families of 64 scenarios.

- [x] **Step 1: Write and run a failing suite-contract test** covering 256 scenarios, task, family sizes, zero-time 2-3 m targets, 2 m route segments, turns, recovery disturbance, 5 cm position gates, and saturation gates.
- [x] **Step 2: Add the TOML suite** with a 2 m forward family, 3 m turn family, three-waypoint turning route, and the same route with a push.
- [x] **Step 3: Run the suite-contract test and existing scenario tests.**

### Task 2: Screen both exact checkpoints

**Files:**
- Read: checkpoint model 100 and protected actor-only transfer under `logs/rsl_rl` and `transfers/`
- Create: immutable evaluation artifacts under `evaluations/`

- [x] **Step 1: Run `ascento evaluate screen` for both checkpoints with batch size 256 on CUDA.**
- [x] **Step 2: Verify checkpoint hashes, suite ID, scenario hash, every gate, and the full per-family results.**
- [x] **Step 3: Run paired evaluation comparison and report the verdict and actionable failure modes.**

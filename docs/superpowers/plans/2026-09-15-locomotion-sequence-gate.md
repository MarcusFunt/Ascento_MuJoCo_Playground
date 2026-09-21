# Locomotion Sequence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-seed evaluator gate that proves recovery, nearby target arrival, and quiet stopping before locomotion promotion.

**Architecture:** Extend the evaluation runner with one evaluator-only relative world-target command and target-phase metrics. Register the binary arrival metric for report aggregation, define the immutable suite, then evaluate the frozen transfer and the 300-iteration candidate with the same scenarios.

**Tech Stack:** Python 3.12, PyTorch, MuJoCo/mjlab, TOML evaluation suites, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-locomotion-sequence-gate.md`

## Global Constraints

- Preserve the original 79,999 balance checkpoint and reject the quiet continuation.
- Do not alter velocity, recovery, jump, waypoint, or terrain task behavior.
- Use evaluator artifacts and gates, not PPO reward, for promotion.

---

### Task 1: Relative world-target evaluator command

**Files:**
- Modify: `src/ascento_mjlab/evaluation/runner.py`
- Test: `tests_mjlab/evaluation/test_runner_lifecycle.py`

**Interfaces:**
- Produces `_apply_world_target_offset(base_env, updates)` for `world_target_offset = (forward_m, lateral_m, yaw_delta_rad)`.
- Consumes the existing world-target state and WXYZ yaw helpers.

- [ ] **Step 1: Write the failing test**

```python
def test_world_target_offset_is_rotated_by_current_yaw():
    # A forward 0.20 m target at +pi/2 yaw becomes +Y in world coordinates.
    ...
    _apply_world_target_offset(env, [(0, (0.20, 0.0, 0.0))])
    assert torch.allclose(world_target_xy(env)[0], torch.tensor([0.0, 0.20]))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -q tests_mjlab/evaluation/test_runner_lifecycle.py::test_world_target_offset_is_rotated_by_current_yaw`

- [ ] **Step 3: Implement the minimal command path**

```python
if name == "world_target_offset":
    _apply_world_target_offset(base_env, updates)
    continue
```

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest -q tests_mjlab/evaluation/test_runner_lifecycle.py::test_world_target_offset_is_rotated_by_current_yaw`

### Task 2: Target-phase evaluator metrics

**Files:**
- Modify: `src/ascento_mjlab/evaluation/runner.py`
- Modify: `src/ascento_mjlab/evaluation/report.py`
- Test: `tests_mjlab/evaluation/test_runner_lifecycle.py`
- Test: `tests_mjlab/evaluation/test_quality_regression.py`

**Interfaces:**
- Produces binary `target_arrived`, numeric `target_arrival_time_s`, `final_target_error`, `post_target_speed_rms`, and `post_target_heading_error_rms` episode metrics.
- `target_arrived` is summarized with Wilson bounds.

- [ ] **Step 1: Write failing aggregation tests**

```python
assert summarize_results(results)["sequence"]["target_arrived"]["success_rate"] == 0.5
```

- [ ] **Step 2: Run tests to verify expected failure**

Run: `pytest -q tests_mjlab/evaluation/test_quality_regression.py`

- [ ] **Step 3: Add target command bookkeeping and final metrics**

```python
target_arrived |= target_commanded & (target_error <= 0.035)
post_target_speed_sq += target_arrived.float() * planar_speed.square()
```

- [ ] **Step 4: Register `target_arrived` as binary and rerun focused tests**

Run: `pytest -q tests_mjlab/evaluation/test_runner_lifecycle.py tests_mjlab/evaluation/test_quality_regression.py`

### Task 3: Immutable suite and regression contract

**Files:**
- Create: `benchmarks/suites/locomotion_sequence_gate_v1.toml`
- Create: `tests_mjlab/evaluation/test_locomotion_sequence_gate.py`
- Modify: `docs/training.md`

**Interfaces:**
- Suite uses a 256-scenario `disturbance` family with 0.05–0.15 m/s pushes and a 5–20 cm `world_target_offset` at 9 s.
- Hard gates cover success/recovery/arrival Wilson LCB; final error, post-target speed/heading p95; and stationary tilt/action-second-difference/rocking p95.

- [ ] **Step 1: Write a suite-load and gate fixture test**

```python
suite = load_suite(Path("benchmarks/suites/locomotion_sequence_gate_v1.toml"))
assert suite.task == "Ascento-Locomotion-Flat"
assert "sequence_target_arrival_lcb" in {gate.gate_id for gate in suite.gates}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest -q tests_mjlab/evaluation/test_locomotion_sequence_gate.py`

- [ ] **Step 3: Create the fixed suite and document its promotion rule**

- [ ] **Step 4: Run the suite test and evaluator unit tests**

Run: `pytest -q tests_mjlab/evaluation/test_locomotion_sequence_gate.py tests_mjlab/evaluation/test_runner_lifecycle.py`

### Task 4: Baseline/candidate evidence and conditional progression

**Files:**
- Create: two directories under `evaluations/locomotion_sequence_gate/`
- Create: `evaluations/locomotion_sequence_gate/model_299_vs_79999.json`

- [ ] **Step 1: Evaluate the frozen actor-only transfer**

Run: `ascento evaluate run --checkpoint <79999-locomotion-transfer> --suite locomotion_sequence_gate_v1 --batch-size 256 --device cuda:0`

- [ ] **Step 2: Evaluate `model_299.pt` on the same suite**

Run: `ascento evaluate run --checkpoint <model_299.pt> --suite locomotion_sequence_gate_v1 --batch-size 256 --device cuda:0`

- [ ] **Step 3: Compare only completed compatible artifacts**

Run: `ascento evaluate compare <transfer-evaluation> <candidate-evaluation> --output locomotion_sequence_gate/model_299_vs_79999.json --json`

- [ ] **Step 4: If any hard gate fails, capture/replay its worst scenario and stop progression; otherwise authorize only the 0.20–1.0 m go-to-pose stage.**

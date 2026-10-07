**Goal:** Add a separate shared-policy locomotion task that expands from short target motion to long waypoint travel, settled stopping, and mild push recovery, then compare it with the existing gate-selected policy.

**Architecture:** Keep the existing `Ascento-Locomotion-Flat` training task contract unchanged. Add a new task with the same actor observation and action topology, expanding target distance from 0.15-0.35 m to 2-3 m and gate-like push/recovery episodes from 10% to 25% over 24,000 control steps. Transfer only actor weights and observation normalization from the selected checkpoint.

**Tech Stack:** Python, mjlab/RSL-RL, PyTorch, and the `ascento` managed-run and evaluator CLI.

**Spec:** `docs/training.md` and the user request in this conversation.

## Global Constraints

- Roadrunner's exact training stages are not public; describe this as an adaptation of its publicly described single-policy, multi-mode approach.
- Preserve the existing locomotion training configuration and signed task contract.
- Preserve the 41-dimensional actor observation topology and six-action interface for explicit actor-only transfer.
- Start from `checkpoints/locomotion_best_gate_20260922/model_01000.pt`; keep critic, optimizer, and iteration state fresh.
- Store run and evaluation artifacts on native WSL ext4 and use the managed CLI.
- Compare only completed evaluations with the same suite and resolved scenarios.

## Review Focus

- Curriculum values clamp at the endpoints; invalid ranges and zero ramp lengths fail clearly.
- The original locomotion task retains its training target and event parameters.
- Actor observations and actions remain compatible for actor-only transfer.
- Recovery exposure increases only through training sampling; evaluation disturbances remain deterministic.
- Both policies are evaluated on the same immutable suite.

### Task 1: Curriculum schedule

**Files:** `src/ascento_mjlab/mdp/events.py`, `tests_mjlab/test_locomotion_curriculum.py`

- [x] Add and test linear interpolation, endpoint clamping, and validation.
- [x] Add opt-in scheduled target-distance and gate-fraction parameters while preserving the existing task defaults.
- [x] Run focused tests.

### Task 2: Additional task

**Files:** `src/ascento_mjlab/tasks/generalist_locomotion/env_cfg.py`, `src/ascento_mjlab/tasks/generalist_locomotion/rl_cfg.py`, `src/ascento_mjlab/tasks/__init__.py`, `dashboard/task_catalog.py`, `tests_mjlab/test_locomotion_curriculum.py`

- [x] Register `Ascento-Generalist-Locomotion-Flat` with 0.15-0.35 m starting goals and 10% gate-like episodes, ramping to 2-3 m and 25% over 24,000 control steps.
- [x] Reuse locomotion PPO settings and register training and play configurations.
- [x] Check registry, schedule, actor ABI compatibility, and dashboard catalog.

### Task 3: Fixed evaluation suite and guidance

**Files:** `benchmarks/suites/roadrunner_generalist_sequence_gate_v1.toml`, `docs/training.md`

- [x] Add 256 deterministic settle, push, short-goal, and stop scenarios with nine hard gates.
- [x] Document the public Roadrunner scope limitation, curriculum, transfer, training, and comparison commands.
- [x] Verify suite discovery and documentation contract.

### Task 4: Train and compare

- [x] Attempt the existing checkpoint on the original locomotion gate; that artifact is INVALID because its task topology is incompatible with the active observations and rewards.
- [x] Create an actor-only transfer baseline and evaluate it on the new suite.
- [x] Train a managed 1,200-iteration CUDA pilot from the transfer with seed `20261007`.
- [x] Evaluate and compare both policies on the same 256 resolved scenarios.
- [x] Record results without treating the pilot as a promoted policy.

## Execution results

- Original locomotion preflight on `locomotion_sequence_gate_v1`: INVALID due task-topology incompatibility after the pushed reward-schema change.
- Actor-only transfer baseline: PASS on all nine gates in `roadrunner_generalist_sequence_gate_v1`.
- Managed pilot `e81d9e5e46b6`: exit 0, 1,200 iterations, 512 environments, CUDA, seed `20261007`; final checkpoint `model_1199.pt`.
- Candidate evaluation `20261007T074109Z_roadrunner_generalist_sequence_gate_v1_model_1199`: FAIL on five hard gates; paired verdict WORSE. It survived all 256 scenarios but arrived at 0/256 targets.
- Paired target-arrival IQM delta: -1.0. Final-target-error IQM delta: +0.16734 m (95% CI +0.16696 to +0.16773 m). Stationary-tilt p95 rose from 0.00945 rad to 0.04492 rad and exceeded the 0.04 rad gate.
- Focused curriculum, task-contract, target-event, and documentation checks: 17 passed. `git diff --check` was clean.
- Do not promote the pilot checkpoint; follow-up reward and progression tuning is needed.

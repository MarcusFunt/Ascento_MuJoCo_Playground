# Reward Alignment Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Compare potential distance shaping and settled completion signals from the same selected locomotion actor while holding the Generalist stage-0 curriculum fixed.

**Architecture:** Keep current rewards as the control. Add an opt-in potential-based progress term whose discount is read from PPO and whose per-transition value is divided by `env.step_dt` to cancel reward-manager dt scaling. Add a bounded, smooth settled-quality term and an optional fixed goal-mix stage, then evaluate each single-feature arm on identical band/cohort scenarios and the existing gate.

**Tech Stack:** Python, PyTorch, pytest, MuJoCo Lab/RSL-RL, Ascento CLI, TOML evaluation suites.

**Spec:** User-provided reward-alignment plan in the current conversation; this plan records implementation details and the controlled comparison it authorizes.

## Global Constraints

- Initialize every arm from `checkpoints/locomotion_best_gate_20260922/model_01000.pt` using `ascento tools initialize-transfer`; keep critic and optimizer fresh.
- Keep `ASCENTO_GENERALIST_OBSTACLE_MODE=0`, precision/recovery cohorts at 0.20/0.20, and navigation goal-mix stage 0 for every arm.
- Change one reward feature per treatment arm; keep safety and posture reward terms fixed.
- For potential shaping, use `k=20`, the PPO config's actual discount, and `F=(gamma*Phi_next-Phi_prev)` with `Phi=-k*distance`.
- For settled quality, use smoothstep scores for distance/heading/speed inside the evaluator envelope and cap paid stable time at 0.35 s per target attempt; the quality-only arm uses weight 8 while the existing one-shot event stays at 5.
- Use the same transfer checkpoint, seed `20261012`, 512 environments, 300 updates, save interval 100, and evaluator scenario hashes across arms.
- Do not stop, restart, or edit files in the canonical checkout while the live run is active; begin GPU experiments only after it finishes.
- Compare settled arrivals, final target-error p95, heading error, arrival time, survival/recovery, by diagnostic-suite family; scalar return is secondary.

## Review Focus

- Per-environment reset must initialize the potential cache from the post-reset robot and target state; target reassignment must not create a spurious distance-jump reward. Test both in Task 1.
- Potential shaping must remain correctly scaled when `step_dt` changes; test dt cancellation and discounted telescoping in Task 1.
- Settled-quality reward must remain bounded per target attempt and must not pay for a 10 cm miss, moving arrival, or bad heading; test those cases in Task 2.
- Fixed-stage mode must prevent online promotion while preserving the current default curriculum when unset; test both modes in Task 1.
- The new suite must give separate summaries for precision, recovery-retarget, and short/medium/long navigation; validate suite families and fixed scenario count in Task 3.

---

### Task 1: Add timestep-correct potential shaping and a fixed-stage control

**Files:**
- Modify: `src/ascento_mjlab/mdp/rewards.py`
- Modify: `src/ascento_mjlab/tasks/generalist_locomotion/env_cfg.py`
- Modify: `src/ascento_mjlab/mdp/events.py`
- Modify: `tests_mjlab/test_reward_contract.py`
- Modify: `tests_mjlab/test_locomotion_curriculum.py`

**Interfaces:**
- Consumes: `world_target_xy`, per-attempt IDs, `AscentoGeneralistLocomotionRlCfg.algorithm.gamma`, and the existing stage-0 goal-mix state.
- Produces: opt-in `ASCENTO_GENERALIST_PROGRESS_REWARD_MODE=potential`, `PotentialBasedTargetDistanceReward`, and `ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE=0`.

- [x] **Step 1: Write failing tests** named `test_potential_target_progress_is_gamma_discounted_and_dt_scaled`, `test_potential_target_progress_resets_and_masks_target_reassignment`, `test_generalist_fixed_goal_mix_stage_freezes_promotion`, and `test_generalist_fixed_goal_mix_stage_override_is_opt_in`. Assert the default velocity reward and adaptive curriculum remain unchanged when overrides are absent.
- [x] **Step 2: Run focused tests and observe the expected failures.**

Run: `env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/.cache/ascento-mjlab/cu128/bin/python -m pytest -q tests_mjlab/test_reward_contract.py tests_mjlab/test_locomotion_curriculum.py -k 'potential_target_progress or fixed_goal_mix_stage'`

Expected: failures identify the missing potential reward/config or fixed-stage behavior after successful test collection.
- [x] **Step 3: Implement the potential reward and explicit optional fixed-stage parameter.** Potential is `-k * distance`; pass PPO's actual gamma; mask target reassignment; return `(gamma * phi_next - phi_prev) / env.step_dt`; keep existing velocity reward as default.
- [x] **Step 4: Run focused tests and verify both override and default behavior.**

Run: `env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/.cache/ascento-mjlab/cu128/bin/python -m pytest -q tests_mjlab/test_reward_contract.py tests_mjlab/test_locomotion_curriculum.py -k 'potential_target_progress or fixed_goal_mix_stage'`

Expected: all selected tests pass.
- [x] **Step 5: Commit** as `feat: add generalist potential progress ablation`.

### Task 2: Add bounded smooth settled-quality credit and bonus override

**Files:**
- Modify: `src/ascento_mjlab/mdp/rewards.py`
- Modify: `src/ascento_mjlab/tasks/generalist_locomotion/env_cfg.py`
- Modify: `src/ascento_mjlab/physics.py`
- Modify: `tests_mjlab/test_reward_contract.py`
- Modify: `scripts/reward_probe.py`

**Interfaces:**
- Consumes: the evaluator waypoint stop envelope (`0.10 m`, `0.20 rad`, `0.08 m/s`, upright/support/posture limits) and existing one-shot settled-stop event.
- Produces: opt-in `ASCENTO_GENERALIST_TARGET_QUALITY_WEIGHT`, bounded `GeneralistTargetSettledQuality`, and a separate `ASCENTO_GENERALIST_TARGET_ACHIEVEMENT_WEIGHT=8` arm.

- [x] **Step 1: Write failing tests** named `test_generalist_settled_quality_is_smooth_inside_evaluator_envelope`, `test_generalist_settled_quality_rejects_miss_motion_and_bad_heading`, `test_generalist_settled_quality_credit_is_capped_per_attempt`, and `test_generalist_target_quality_and_success_bonus_weights_are_independent`. Assert zero quality at 10 cm, with speed above 0.08 m/s, or heading above 0.20 rad; assert full credit is bounded to one 0.35 s window.
- [x] **Step 2: Run focused tests and observe the expected failures.**

Run: `env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/.cache/ascento-mjlab/cu128/bin/python -m pytest -q tests_mjlab/test_reward_contract.py -k 'generalist_settled_quality or target_quality_and_success_bonus'`

Expected: failures identify the missing quality term/config, not test collection/import errors.
- [x] **Step 3: Implement smooth quality credit, cap it to one held window per target, preserve the existing one-shot event, and bump the reward schema.** Keep safety/posture weights unchanged.
- [x] **Step 4: Extend the deterministic reward probe to assert settled success outranks the miss, moving arrival, and bad-heading arrival under each relevant arm.**
- [x] **Step 5: Run reward tests and the probe; commit** as `feat: add bounded settled quality reward ablation`.

Run: `env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/.cache/ascento-mjlab/cu128/bin/python -m pytest -q tests_mjlab/test_reward_contract.py && env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/.cache/ascento-mjlab/cu128/bin/python scripts/reward_probe.py`

Expected: focused tests pass and the probe reports settled success as the top-ranked case for the control, bonus-8, and quality arms.

### Task 3: Define fixed-seed band/cohort evaluation and record the experiment

**Files:**
- Create: `benchmarks/suites/generalist_reward_alignment_v1.toml`
- Create: `docs/experiments/reward_alignment_v1.json`
- Modify: `docs/training.md`
- Modify: evaluation-suite tests under `tests_mjlab/`

**Interfaces:**
- Consumes: the policy output of Task 1/2 and the evaluator's per-family target, heading, survival, recovery, and arrival-time metrics.
- Produces: fixed families `precision_short`, `recovery_retarget_short`, `navigation_short`, `navigation_medium`, and `navigation_long`, each with 64 scenarios; experiment matrix arms `control`, `bonus_8`, `settled_quality`, and `potential_progress`.

- [x] **Step 1: Write failing suite/config test** `test_generalist_reward_alignment_suite_has_fixed_band_cohort_families` for five family labels, 320 total scenarios, fixed root seed, and diagnostic-only gates.
- [x] **Step 2: Run the tests and observe the expected failure.**

Run: `env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/.cache/ascento-mjlab/cu128/bin/python -m pytest -q tests_mjlab/evaluation/test_scenarios.py -k generalist_reward_alignment_suite`

Expected: the test cannot find/materialize the new suite; no unrelated test failure.
- [x] **Step 3: Add the suite, experiment protocol, arm environment overrides, and reproducible CLI instructions.** Use the existing `roadrunner_generalist_sequence_gate_v1` as the strict gate alongside the new family-sliced diagnostic suite.
- [x] **Step 4: Run suite parsing/materialization tests and update docs; commit** as `docs: add reward alignment ablation protocol`.

Run: `env CUDA_VISIBLE_DEVICES= PYTHONPATH=src $HOME/.cache/ascento-mjlab/cu128/bin/python -m pytest -q tests_mjlab/evaluation/test_scenarios.py -k generalist_reward_alignment_suite`

Expected: 320 deterministic scenarios are materialized across the five named families.

### Task 4: Run matched transfer experiments and evaluate outcomes

**Files:**
- Update: `docs/experiments/reward_alignment_v1.json` with actual run IDs, checkpoint hashes, evaluation IDs, paired deltas, and conclusions.

**Interfaces:**
- Consumes: the selected actor-only transfer and the four control/treatment configurations from Tasks 1–3.
- Produces: deterministic gate and family-sliced evaluation reports for the transfer and each 300-update run.

- [ ] Wait for live run `bd2730714479` to finish; verify it ended naturally and leave its artifacts intact.
- [ ] Create a fresh Generalist transfer from the selected locomotion checkpoint, evaluate the transfer on both suites, and verify obstacle mode is zero.
- [ ] Run the control, bonus-8-only, settled-quality-only, and potential-progress-only arms sequentially from that same transfer, seed `20261012`, 512 environments, 300 updates, saving every 100 updates; keep the goal mix at stage 0.
- [ ] Evaluate update 100, 200, and 300 checkpoints on `roadrunner_generalist_sequence_gate_v1`; evaluate the transfer and update-300 checkpoints on `generalist_reward_alignment_v1`.
- [ ] Record settled arrivals, p95 target error, heading error, arrival time, survival/recovery, cohort/band family, return, and run lineage. State whether any arm improves outcomes without losing the existing gate.
- [ ] Commit the completed experiment record as `docs: record reward alignment ablation results`.

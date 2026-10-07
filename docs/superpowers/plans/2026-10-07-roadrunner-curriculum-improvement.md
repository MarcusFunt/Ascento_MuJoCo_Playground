**Goal:** Improve the Roadrunner-inspired generalist locomotion fine-tune while preserving the selected transfer policy's target-arrival, heading, and quiet-stop quality.

**Status:** Bounded implementation, training, and evaluation are complete. Frozen-normalizer model_99 passes all nine gates and the 0.05 m practical target but is NOT_PROVEN_BETTER against adaptive BC and the actor-only transfer. Keep the transfer selected. The normalizer ablation is complete. Training recovery and task-slice telemetry still need lifecycle/cohort fixes before further curriculum or reward conclusions are trusted; the fixed-share and short-target reward ablations remain open.

**Architecture:** Keep the existing fixed sequence suite as the promotion gate. Preserve a short-goal/recovery task slice throughout training, report training success by task slice, constrain how far PPO can move the transferred policy, and evaluate saved checkpoints during training. Promote only a checkpoint that passes all hard gates and demonstrates a paired improvement over the transfer baseline.

**Tech Stack:** Python, mjlab/RSL-RL, PyTorch, the managed `ascento` CLI, and the existing SQLite-backed evaluation reports.

**Spec:** `docs/training.md`, `benchmarks/suites/roadrunner_generalist_sequence_gate_v1.toml`, and the user request to improve the Roadrunner-inspired pilot.

## Run analysis

The successful managed run was a fine-tune from the actor-only transfer checkpoint, not a fresh policy. Run `e81d9e5e46b6` used 512 environments, seed `20261007`, and 1,200 iterations. It started from the same transfer artifact evaluated as the baseline, then continued PPO updates for about 24 minutes. The transfer baseline passed all nine hard gates on the 256 fixed scenarios.

The final checkpoint `model_1199.pt` completed without invalid/non-finite updates, but its same-suite comparison was **WORSE** than the transfer baseline. It survived all 256 scenarios and recovered from all scripted pushes, but reached the commanded target in **0/256**. Final-target-error p95 was **0.1753 m** (gate: 0.10 m); stationary-tilt p95 was **0.04492 rad** (gate: 0.04 rad). Since it never arrived, post-target speed and heading could not be measured. Action second-difference and body-rocking gates passed. The paired IQM deltas were -1.0 for target arrival, +0.16734 m for final target error (95% CI +0.16696 to +0.16773 m), and +0.03525 rad for stationary tilt (95% CI +0.03516 to +0.03534 rad). The final policy therefore remained upright but did not execute the short target-and-stop sequence reliably.

Evaluation of saved checkpoints locates the regression:

| Policy checkpoint | Suite result | Target arrival | Final target error p95 | Post-target heading p95 | Stationary rocking p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Actor-only transfer | PASS | 256/256 | 0.00557 m | 0.0442 rad | 0.0150 |
| `model_250.pt` | PASS | 256/256 | 0.00911 m | 0.0476 rad | 0.0204 |
| `model_500.pt` | FAIL | 256/256 | 0.04909 m | 0.2840 rad | 0.0382 |
| `model_750.pt` | FAIL | 256/256 | 0.01658 m | 0.3466 rad | 0.0513 |
| `model_1000.pt` | FAIL | 256/256 | 0.03982 m | 0.2903 rad | 0.0379 |
| `model_1199.pt` | FAIL | 0/256 | 0.17529 m | not measurable | 0.0271 |
| BC `model_100.pt` | PASS | 256/256 | 0.01818 m | 0.01642 rad | 0.01640 |
| BC `model_199.pt` | formal PASS, practical target FAIL | 256/256 | 0.08746 m | 0.06777 rad | 0.02894 |

The first vanilla intermediate checkpoint passes every gate, but the paired comparison against the transfer policy is NOT_PROVEN_BETTER: model 250 increased final-error IQM by 0.00380 m, stationary-tilt IQM by 0.01171 rad, and body-rocking IQM by 0.00495. The BC model 100 also passes all formal gates and stays below the user's 0.05 m practical error target, but its paired final-error IQM is +0.01196 m; model 199 degrades to 0.08746 m p95 and is outside the practical target. Checkpoints 500, 750, and 1,000 still arrive, yet fail post-target heading (limit 0.15 rad) and stationary body rocking (limit 0.03). From model 250 to model 500, paired final-error IQM rose 0.03742 m, post-target heading IQM rose 0.23583 rad, and body-rocking IQM rose 0.01501. After iteration 1,000, the final checkpoint also loses target arrival and fails final error and tilt. The failures are therefore not explained by a broken evaluator or an immediate bad transfer: they appear during continued fine-tuning and expand later.

Trainer metrics reinforce the need for checkpoint selection by task performance. Mean training reward went from 131.7 at iteration 100 to 272.8 at 250, then 77.8 at 500, 46.1 at 750, 63.8 at 1,000, 56.5 at 1,100, and 134.3 at 1,199. Episode length and fall counts also moved substantially during training. The late reward rebound did not correspond to recovery on the fixed target sequence. The run reported no invalid PPO updates, so this was a policy-quality regression rather than a numerical crash.

### Likely causes to test

1. **The task mix shifted away from the exact behavior the gate measures.** Target distances ramp from 0.15–0.35 m to 2–3 m over 24,000 control steps (about 1,000 iterations at 24 rollout steps per environment). The gate-like episode fraction rises only from 10% to 25%, leaving most experience focused on long waypoint travel. The gate instead commands a 0.15 m target at 9 seconds and checks precise arrival, heading retention, and stopping. The timeline—full-gate performance at 250, then heading/rocking failures by 500 and zero arrival by 1,199—is consistent with loss of short-range and post-target behavior as fine-tuning proceeds.

2. **The reward is only an approximation of the gate.** Training rewards target progress and smooth proximity, penalizes speed near a target, and resamples normal waypoints after a 0.04 m arrival plus dwell. It does not log a training success rate that directly mirrors the evaluator's 0.035 m arrival test and post-target heading/rocking checks. The measured gate failures show this gap; whether the reward weights themselves cause it remains to be established by ablation.

3. **PPO and observation-normalizer drift may amplify forgetting.** The run used adaptive PPO with a 1e-4 initial learning rate and continued for 1,200 updates from the passing transfer. The target-error distribution also widened during the curriculum. The evidence does not isolate policy updates from normalizer movement, so the next experiment should measure both and change one at a time.

## Training telemetry validity caveat

The fixed-suite evaluation results and paired policy comparisons are valid, but two training telemetry paths are not yet suitable for learning claims. In src/ascento_mjlab/mdp/events.py, retarget clears recovery_completed, then the recovery detector excludes already-retargeted environments. A successful pre-retarget recovery is therefore lost before episode completion, making the training gate-recovery LCB unreliable and unsafe as a stage-advancement signal. Separately, short/medium/long masks include gate-like episodes, and the metrics do not distinguish receiving a push, recovery, retarget, the second arrival, and a settled stop. Correct and test those paths before using training recovery LCB or overlapping slice outcomes to claim behavior was learned.

For frozen run 0d0d2f9dbb55, model_99 reached progress 0.0996 and scheduled gate-like share 0.1149 (25% is the configured endpoint). Its goal-mix stage remained 0; only 42 gate-like episodes had completed, and short-arrival LCB was 0.161. The recovery LCB was 0 but is uninterpretable for the lifecycle reason above.

## Global constraints

- Keep the actor-only transfer checkpoint as the quality floor and rollback policy.
- Do not promote `model_1199.pt`; it fails the fixed-suite gates.
- Use the same suite definition, seed, resolved scenarios, deterministic inference, and gate thresholds for every checkpoint comparison.
- Treat training reward as diagnostic only; it cannot override a failed hard gate.
- Use the project owner's practical final-target-error p95 limit of 0.05 m; the fixed suite's formal hard gate remains 0.10 m.
- Change one curriculum, reward, or optimizer factor per ablation so results remain interpretable.
- Roadrunner's private curriculum details remain unavailable; this plan evaluates the project's adaptation, not an exact reproduction.

## Review focus

- Short-goal, long-goal, push-recovery, and retarget/stop exposure is reported separately.
- Curriculum progress and sampled target distances are observable at each checkpoint.
- Transferred actor normalization statistics and their drift are recorded.
- Any checkpoint used as a candidate has a complete evaluation report and matching resolved-scenario hash.
- A candidate must pass every hard gate and show a paired improvement before promotion.

### Task 1: Make the regression visible during training

**Files:** `src/ascento_mjlab/mdp/events.py`, `src/ascento_mjlab/tasks/generalist_locomotion/env_cfg.py`, `tests_mjlab/test_locomotion_curriculum.py`

- [x] Record curriculum step/fraction, sampled target-distance bands, and gate-like episode counts in managed training metrics.
- [x] Add initial per-slice measures for arrival within 0.035 m, recovery, heading at arrival, and fall/timeout rate.
- [x] Add tests for metric/reset behavior and for the configured distribution at curriculum start, midpoint, and endpoint.
- [ ] Make the reported goal slices disjoint from gate-like episodes; track push received, successful recovery through retarget, second-target arrival, settled stop, and post-arrival heading separately. Add a lifecycle test showing successful recovery remains recorded through retarget until episode reset.
- [x] Verify that run telemetry exposes these metrics at the saved checkpoint cadence.

### Task 2: Preserve precision tasks while expanding travel

**Files:** `src/ascento_mjlab/tasks/generalist_locomotion/env_cfg.py`, `src/ascento_mjlab/mdp/events.py`, `src/ascento_mjlab/mdp/rewards.py`, `tests_mjlab/test_locomotion_curriculum.py`, `tests_mjlab/test_reward_contract.py`

- [x] Replace the single expanding target-distance distribution with a stratified mixture that keeps short-goal and gate-like episodes present at every stage while adding long waypoint tasks.
- [x] Compare gate-like exposure levels (including the current 25% endpoint) while holding optimizer settings constant.
- [ ] Audit reward contributions on short-target success and near-target failure cases. Test an explicit arrival/settle and target-heading objective only if the logged reward breakdown shows the current shaping under-rewards those outcomes.
- [x] Use gate success thresholds to decide when the long-distance share may increase; do not advance solely because a fixed control-step count elapsed.

### Task 3: Limit transfer-policy drift

**Files:** `src/ascento_mjlab/tasks/generalist_locomotion/rl_cfg.py`, training configuration, and focused PPO/normalizer tests

- [x] Add an optimizer ablation with a lower update rate and tighter KL limit, keeping the task mixture unchanged.
- [x] Persist target-error observation-normalizer mean/scale for each screened checkpoint and compare adaptive versus preserved transfer statistics; frozen model_99 kept them unchanged but was not promoted.
- [x] If smaller updates still erase gate behavior, test a reference-policy behavior-cloning term against the frozen transfer actor.
- [x] Keep each stabilization method as a separate run so its contribution is measurable.

### Task 4: Retrain with quality-based checkpoint selection

**Files:** managed training commands and `docs/training.md`

- [x] Start every pilot from the same actor-only transfer checkpoint and record its SHA-256, seed, task config, and curriculum settings.
- [x] Save and screen checkpoints every 100–250 iterations on `roadrunner_generalist_sequence_gate_v1`; compare all complete checkpoints with the transfer baseline.
- [x] Stop or roll back at the first sustained hard-gate regression instead of continuing to 1,200 iterations based on reward rebound.
- [ ] Run a second seed only after one seed preserves all hard gates and shows an acceptable paired quality tradeoff; run the full authoritative suite on the selected checkpoint before promotion.

### Task 5: Acceptance criteria

- [ ] A selected checkpoint passes all nine hard gates, including target-arrival Wilson lower bound >= 0.50, final-target-error p95 <= 0.05 m as the practical target (formal gate <= 0.10 m), post-target heading p95 <= 0.15 rad, stationary tilt p95 <= 0.04 rad, and body-rocking p95 <= 0.03. BC model 100 meets this absolute screen but lacks an acceptable paired quality tradeoff; model 199 exceeds the practical target.
- [ ] Paired evaluation against the transfer policy uses 256 matching resolved scenarios. Require all hard gates, no statistically credible regression on guarded quality metrics, and a prespecified paired improvement whose confidence interval excludes zero; a PASS alone does not establish an upgrade.
- [ ] Disjoint short-goal, long-goal, push/recovery, retarget, second-arrival, and settled-stop metrics show that each intended behavior was sampled and learned; recovery success remains recorded through retarget until episode reset.
- [ ] A candidate's performance is repeatable across the second seed before the task documentation calls it an improvement.

## Verification plan

- Run focused tests for the modified curriculum, reward contracts, and fixed-suite evaluator.
- Run bounded 300–500 iteration pilots, inspect managed telemetry, and screen every complete saved checkpoint. Record any CUDA/runtime interruption separately from policy-quality outcomes.
- Use `ascento evaluate compare` against the transfer checkpoint for selected complete evaluations and run the 256-scenario authoritative suite before promotion.
- Run the full project suite before integrating the implementation. The current full-suite baseline has one unrelated existing mismatch: `tests_mjlab/test_task_config.py::test_quiet_balance_is_versioned_and_uses_horizon_curriculum` expects `-0.50`, while the unchanged quiet-balance config on `origin/main` sets `-10.0`.

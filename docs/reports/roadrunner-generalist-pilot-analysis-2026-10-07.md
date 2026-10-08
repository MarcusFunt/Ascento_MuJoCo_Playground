# Roadrunner-inspired curriculum pilot analysis

**Date:** 2026-10-07
**Practical target precision:** final-target-error p95 at or below **0.05 m**, per the project owner's clarification. The fixed suite's formal hard gate remains **0.10 m**.
**Decision:** keep the actor-only transfer as the selected policy. The frozen-reference BC and frozen-normalizer checkpoints pass the absolute gates and 0.05 m target-error screen, but neither is a proven upgrade under paired quality comparisons.

## Executive summary

The earlier fine-tuning pilots show that higher training reward and a larger gate-like share do not reliably preserve the transfer policy's target-and-stop behavior. A lower-learning-rate/tighter-KL arm also regressed, so optimizer size alone does not explain the loss. The success-gated long-goal mix remained at stage 0. In the BC continuation, the short-arrival Wilson lower bound was about 0.20, below its 0.50 threshold. The training gate-recovery lower bound was 0, but that signal is not interpretable: the existing retarget callback clears `recovery_completed`, then excludes retargeted environments from further recovery tracking. So the telemetry does not establish that the policy failed to learn recovery; the recovery counter needs a separate lifecycle fix before it can govern or diagnose curriculum advancement.

A frozen-reference behavior-cloning (BC) term was added as a separate, opt-in stabilization arm. Its checkpoint at iteration 100 passed all nine suite gates, arrived at the target in 256/256 scenarios, and had final-target-error p95 **0.01818 m**, comfortably within the requested 0.05 m tolerance. Paired evaluation showed a statistically clear target-error regression against the transfer baseline (IQM +0.01196 m, 95% CI +0.01177 to +0.01217 m), offset by better heading and stationary tilt, with small in-gate regressions in action smoothness and rocking. It was **PASS / NOT_PROVEN_BETTER**.

A matched frozen-normalizer arm completed 100 PPO iterations from the same
transfer used by adaptive BC. Its terminal checkpoint passed all nine gates
with target-error p95 **0.01523 m**. The target-error normalizer buffers stayed
identical to transfer statistics, while the adaptive BC checkpoint's target
axes expanded by 1.22x and 1.40x. Frozen normalization improved paired
target-error IQM versus adaptive BC, but worsened heading and stationary tilt;
versus the selected transfer, target-error IQM regressed by 0.00932 m. The
paired verdict was NOT_PROVEN_BETTER against both controls, so the transfer
remains selected.

The continuation's first attempt stopped at iteration 158 with CUDA 999 surfaced inside an unrelated leg-pose reward and no new checkpoint. An exact same-settings retry completed through iteration 199. Its checkpoint passed all nine formal gates, but target-error p95 rose to **0.08746 m**, outside the requested 0.05 m tolerance, and it was significantly worse than both iteration 100 and the transfer on target error, post-target heading, stationary tilt, and body rocking. Neither checkpoint is promoted.

## Evaluation setup

- Fixed suite: `roadrunner_generalist_sequence_gate_v1`, 256 resolved scenarios, scenario SHA-256 `a7afcfe432d11d1bcc7694faf4e499ae9a62f37244b86a9a9e4b0de790892fc1`.
- Baseline: signed actor-only transfer `transfers/ascento_generalist_locomotion_flat/roadrunner_generalist_signed/model_000000.pt`; evaluation `20261007T085540Z_roadrunner_generalist_sequence_gate_v1_model_000000`; all nine gates pass, target-error p95 0.00557 m.
- BC candidate: run `c93a9c51dff2`, checkpoint `model_100.pt`, SHA-256 `c356bb95e2d17c12f809dd370ccfb2393643b7fddba278cbd5d6743ff5f090ad`; evaluation `20261007T104927Z_roadrunner_generalist_sequence_gate_v1_model_100`. The candidate and baseline used the same suite/scenario identity and passed report consistency checks.
- Frozen-normalizer arm: run 0d0d2f9dbb55, 100 PPO iterations, terminal checkpoint
  model_99.pt (zero-based checkpoint index), SHA-256
  38be3531f3607cbd514f933648499c5103eee90a47f11571989dcde06d2132de; evaluation
  20261007T120309Z_roadrunner_generalist_sequence_gate_v1_model_99.
- All three evaluations used the same 256 resolved scenarios with SHA-256
  a7afcfe432d11d1bcc7694faf4e499ae9a62f37244b86a9a9e4b0de790892fc1.
- Training configuration: 512 environments, seed 20261007, PPO defaults, a 25% scheduled gate-like endpoint, frozen teacher copied from the same actor-only transfer, BC weight 2.0. By model_99 the 100-update screen was only 9.96% through the schedule, at a scheduled gate-like share of 11.49%; it did not train at the 25% endpoint. Training and evaluation used the same signed task contract and BC environment settings.

## Checkpoint comparisons

| Experiment | Checkpoint | Suite / target error p95 | Other observed result |
| --- | ---: | --- | --- |
| Actor-only transfer | — | PASS / 0.00557 m | 256/256 arrivals; quality floor |
| 25% gate-like, default PPO | 100 | PASS / 0.00900 m | Paired result NOT_PROVEN_BETTER |
| 25% gate-like, default PPO | 200 | FAIL / 0.0529 m | Heading 0.162 rad and rocking 0.038 fail |
| 25% gate-like, default PPO | 300 | PASS / 0.05856 m | Formal gates pass; above owner's 0.05 m precision target; still NOT_PROVEN_BETTER |
| 25% gate-like, default PPO | 400 | FAIL / 0.0355 m | Heading 0.280 rad and rocking 0.0477 fail |
| 25% gate-like, default PPO | 499 | FAIL / 0.0848 m | Arrival, heading, and rocking regress |
| 40% gate-like mix | 100 | PASS / 0.00562 m | Paired result NOT_PROVEN_BETTER |
| 40% gate-like mix | 200 | FAIL / 0.0697 m | Rocking 0.0319 fails |
| 40% gate-like mix | 300 | FAIL / 0.1253 m | Target error and tilt fail |
| 40% gate-like mix | 400 | FAIL / 0.1436 m | Heading and target error fail |
| Conservative PPO (3e-5 LR, desired KL 0.003) | 100 | PASS / 0.01758 m | Paired result NOT_PROVEN_BETTER |
| Conservative PPO | 200 | FAIL / 0.1070 m | Target error and heading fail |
| Conservative PPO | 300 / 400 | FAIL / 0.1984 / 0.2397 m | Drift continues; at 400 heading 0.336 rad, action jitter 0.429, rocking 0.148 |
| Frozen-reference BC, weight 2 | 100 | PASS / 0.01818 m | 256/256 arrivals; all nine hard gates pass; paired result NOT_PROVEN_BETTER |
| Frozen-reference BC, weight 2 | 199 | PASS / 0.08746 m | All nine formal gates pass, but above 0.05 m practical target; paired worse than iteration 100 and transfer |
| Frozen-transfer actor normalizer, BC weight 2 | 99 | PASS / 0.01523 m | All gates pass and within 0.05 m; NOT_PROVEN_BETTER vs adaptive BC and transfer |

The 25% and 40% comparisons show that increasing exposure to the gate-like task alone did not prevent forgetting. Smaller PPO updates also did not prevent regression by checkpoint 200. Training reward was not a dependable selection metric: it rebounded late in a run whose fixed-suite target behavior had already degraded.

## BC checkpoint detail

At iteration 100, the candidate's p95 values were: target error 0.01818 m; post-target speed 0.01771 m/s; post-target heading error 0.01642 rad; stationary tilt 0.00422 rad; stationary action second-difference 0.02242; and stationary body rocking 0.01640 rad. All are within the fixed suite's limits.

At iteration 199, all formal gates still passed, but the target-error p95 was 0.08746 m and body-rocking p95 was 0.02894 rad, leaving little margin. On the same paired scenarios, model 199 versus model 100 increased target-error IQM by 0.06803 m (95% CI +0.06773 to +0.06831), post-target heading by 0.05202 rad, stationary tilt by 0.02069 rad, and body rocking by 0.00825 rad. Model 199's final-target-error IQM was +0.07985 m versus the transfer baseline. It fails the owner's practical 5 cm target even though it passes the suite's 10 cm formal gate.

Against the transfer on the 256 paired scenarios:

- Final-target-error IQM increased by 0.01196 m (95% CI +0.01177 to +0.01217). Candidate p95 remains 0.03182 m below the user's 0.05 m limit.
- Post-target heading-error IQM improved by 0.02557 rad (95% CI −0.02623 to −0.02491).
- Stationary-tilt IQM improved by 0.00488 rad (95% CI −0.00496 to −0.00480).
- Body-rocking IQM increased by 0.00125 rad (95% CI +0.00112 to +0.00137); action second-difference IQM also increased by 0.00477. Both still pass their gates.
- The candidate's mean absolute commanded effort increased by about 1.18 Nm in paired IQM.

These are meaningful tradeoffs, not a general improvement. A second seed was not started because iteration 100 did not prove an upgrade and iteration 199 exceeded the owner's practical 5 cm target.

## Frozen-normalizer ablation

The managed run 0d0d2f9dbb55 used the same transfer checkpoint, BC teacher and
weight, 25% scheduled gate-like endpoint, seed 20261007, 512 environments, and
default PPO settings as the adaptive BC arm. It completed 100 PPO iterations
without invalid updates; RSL-RL names the terminal checkpoint model_99.pt
because its checkpoint indices are zero-based. The accepted 100-update screen
is one optimizer update shorter than adaptive BC model_100: checkpoint normalizer
counts show 100 updates for the frozen arm and 101 for adaptive BC. Treat their
paired deltas as a close but not exactly update-matched comparison. The
normalizer summary and paired comparisons are saved as normalizer_drift.json,
paired_comparison.json, and paired_vs_transfer.json under logs/rsl_rl/20261007_135539_roadrunner-frozen-actor-normalizer-bc-w2_0d0d2f9d.

At model_99, curriculum progress was 0.0996 and the scheduled gate-like share
was 0.1149. The success-gated long-goal stage remained at 0 with short/medium/
long shares 0.65/0.25/0.10; only 42 gate-like episodes had completed (below the
64-episode minimum), and short-arrival LCB was 0.161 (below 0.50). Training
gate-recovery LCB was 0, but the recovery telemetry lifecycle described above
makes that value inconclusive. The paired evaluation independently measured
recovery and passed its recovery gate in 256/256 scenarios.

The frozen actor's target-error mean, scale, and count remained exactly equal
to transfer statistics through model_99. In adaptive BC model_100, the two
target-error scale ratios were 1.219 and 1.402, with mean deltas +0.00080 m and
+0.00264 m. This confirms the normalization change occurred; it does not prove
that it caused the quality changes.

| Checkpoint | Target-error p95 | Post-target heading p95 | Stationary tilt p95 | Action-second-difference gate p95 | Body-rocking p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive BC model_100 | 0.01818 m | 0.01642 rad | 0.00422 rad | 0.02242 | 0.01640 |
| Frozen-normalizer model_99 | 0.01523 m | 0.02138 rad | 0.00590 rad | 0.02091 | 0.01534 |

Against adaptive BC, frozen normalization reduced target-error IQM by
0.00268 m (95% CI -0.00276 to -0.00260), but increased post-target heading
IQM by 0.00498 rad, stationary-tilt IQM by 0.00156 rad, post-target speed IQM
by 0.00168 m/s, and action-rate RMS IQM by 0.00372. Stationary action
second-difference IQM improved by 0.00234 (paired delta -0.00234) and stationary
body-rocking IQM by 0.00073, while whole-episode action-second-difference IQM
worsened by 0.00605.
All absolute hard gates still passed.

Against the actor-only transfer, frozen-normalizer model_99 increased
target-error IQM by 0.00932 m (95% CI +0.00917 to +0.00947), while improving
heading and stationary tilt. It also increased whole-episode and stationary
action second-difference IQM by 0.00388 and 0.00259, respectively, stationary
body-rocking IQM by 0.00050, post-target speed IQM by 0.00182 m/s,
and effort RMS IQM by 1.099. Target-error p95 remains within the practical
0.05 m limit, but the mixed paired outcomes do not justify promotion. Both
paired comparison artifacts report NOT_PROVEN_BETTER.

## Root-cause findings and limits

1. **Curriculum telemetry is not yet sufficient for recovery conclusions.** In the BC continuation, progress was about 0.15, scheduled gate-like share about 0.1225, gate-recovery LCB 0, and short-arrival LCB about 0.20. The goal-mix stage remained 0 with approximately 0.65/0.25/0.10 short/medium/long shares. The low short-arrival LCB is evidence that this training slice had not met its advancement threshold. The zero gate-recovery LCB is not evidence of failed recovery because retarget clears the training recovery flag before episode completion. Also, current short/medium/long masks overlap gate-like episodes, and the metrics do not separately count push, retarget, post-retarget arrival, and settled stop. Fix these training measures before using them to decide whether those behaviors were learned. The fixed-suite recovery measurements remain valid because the evaluator computes them separately.
2. **More gate-like episodes were insufficient.** Both 25% and 40% arms first failed formal gates at checkpoint 200; target-error p95 was worse in the 40% arm (0.0697 m vs. 0.0529 m). Episode frequency alone does not teach the exact post-push retarget, arrival, and quiet-stop sequence.
3. **Reducing PPO step size was insufficient.** The conservative arm still failed by checkpoint 200, so optimizer drift is not the only likely cause.
4. **The BC term is not yet calibrated as a reliable trust region.** It yielded an iteration-100 checkpoint inside the 5 cm target, but target-error IQM was already worse than baseline. During the first continuation attempt the per-step BC reward reached about −0.0105, and on the successful retry it reached about −0.0751 near iteration 199. The iteration-199 policy's target p95 and other paired metrics were substantially worse. This indicates increasing divergence from teacher actions; it does not isolate whether weight, student-normalizer drift, or states outside the teacher's distribution caused it.
5. **Normalizer drift is measured, but freezing it is not a complete quality fix.** Adaptive model_100 target-error scales were 1.219x and 1.402x transfer values; frozen model_99 exactly retained transfer statistics and modestly improved target-error IQM versus adaptive BC. It nevertheless worsened heading, tilt, and post-target speed versus adaptive BC and regressed target-error IQM versus transfer. Drift may contribute to target precision, but the ablation does not isolate it as the sole cause.
6. **There is a separate GPU reliability issue.** Earlier BC launch attempts exposed a CPU-only PyTorch environment and an asynchronous Warp/CUDA 999 error. Adding a targeted CUDA synchronization let the parent BC run reach iteration 196 and its iteration-100 evaluation pass. One continuation failed at iteration 158 inside `leg_pose_hold_penalty`; the exact retry completed through iteration 199 and evaluated with no CUDA error. No PPO update was invalid in either continuation, so the failure is runtime, not policy quality. A prior evaluation also failed while sharing the GPU with a trainer; the isolated reevaluation passed.

The experiments support curriculum/reward mismatch and observation-normalizer drift as hypotheses. They do not prove either is the sole cause.

## Recommended next experiments

1. Keep the signed actor-only transfer as the deployed/default policy and fixed-suite baseline.
2. Preserve the 5 cm practical final-target-error threshold and 10 cm formal suite gate. Screen every checkpoint on the same 256 scenarios; report both paired deltas and absolute target p95.
3. The normalizer report and adaptive-versus-frozen ablation are complete. Keep frozen statistics as an experimental control; neither profile is a selected-policy upgrade.
4. Fix the training recovery lifecycle and split the short/long and gate-like outcome cohorts. Preserve a successful recovery flag through retarget until episode reset; count push received, stable recovery, retarget, post-retarget arrival, settled stop, and post-arrival heading separately. Add lifecycle tests and verify the counters against fixed logged trajectories before using them for curriculum decisions.
5. After those metrics are trustworthy, test a fixed 25% gate-like share from the start against the current 10%-to-25% schedule, holding the long-goal mix and PPO settings constant.
6. Audit short-task reward contributions by term for episodes that arrive within 0.035 m and settle with the requested heading. If those outcomes are under-rewarded, add an explicit arrival/settle term and test it alone.
7. Recalibrate reference regularization in a separate sweep (for example, a lower BC weight and a slow schedule tied to gate-like episodes). First demonstrate that it preserves target p95 within 0.05 m and does not worsen guarded rocking/action metrics; do not infer success from return.
8. For each arm, save/screen at 50–100 iteration intervals initially, stop on the first sustained gate failure, and do not start a second seed until a candidate passes all gates and shows an acceptable paired tradeoff.
9. Run only one CUDA workload at a time. Keep `--extra cu128` active, verify the resolved PyTorch CUDA device before launch, record the first CUDA exception separately from cleanup errors, and use `CUDA_LAUNCH_BLOCKING=1` only for bounded diagnosis because it cut throughput to roughly one quarter.

## Verification

The focused generalist-normalizer tests passed (5 tests), and Ruff passed on
the changed Python files. The full suite reported 409 passed and one known,
unrelated mismatch:
tests_mjlab/test_task_config.py::test_quiet_balance_is_versioned_and_uses_horizon_curriculum
expects reward weight -0.50 while the unchanged quiet-balance configuration
sets -10.0. The frozen-normalizer training run exited 0, and all nine fixed
suite gates passed for the terminal checkpoint.
## Guard generalist locomotion roadmap WP-00 baseline freeze

- Code commit evaluated: `49e6e89c8237749ea7586c23016e35965a58bcbc`. The checkout was dirty only in the six unrelated maintenance paths listed in `docs/experiments/ascento_guard_generalist_locomotion_v3.json`.
- Selected rollback policy: `transfers/ascento_generalist_locomotion_flat/roadrunner_generalist_signed/model_000000.pt`, SHA-256 `04e6e121faf1dec19376c33448ef75aa280b2b7448ccc674f511e7b79f332959`.
- Development suite: `roadrunner_generalist_sequence_gate_v1`, 256 scenarios, suite-definition SHA-256 `7e38369b336b1283d2d2c1051083d1668247e9d4b8489a0dd91c4811b06d10da`, resolved-scenario SHA-256 `a7afcfe432d11d1bcc7694faf4e499ae9a62f37244b86a9a9e4b0de790892fc1`.
- Current-code baseline evaluation: `20261007T161244Z_roadrunner_generalist_sequence_gate_v1_model_000000` — PASS, 256/256 target arrivals, final-target-error p95 `0.0055662624 m`. Paired against the prior baseline report on all 256 scenarios, the arrival delta was zero and target-error IQM changed by `-0.0000003301 m`.
- Frozen promotion suite: `roadrunner_generalist_sequence_promotion_v1`, root seed `2026100701`, 256 scenarios, definition SHA-256 recorded in the experiment manifest. It uses the same gates and command/reset/disturbance semantics as the development suite. It is reserved for the final selected checkpoint.
- The machine-readable experiment matrix and promotion-use restriction are in `docs/experiments/ascento_guard_generalist_locomotion_v3.json`.

## Guard generalist locomotion roadmap WP-02 morphology baseline

The selected transfer was screened on all four diagnostic families before any morphology thresholds or reward penalties were introduced. The run used the exact selected checkpoint SHA-256, the matching frozen task topology, and commit `49e6e89c8237749ea7586c23016e35965a58bcbc` with only the diagnostic suite and evaluator instrumentation present in the detached baseline checkout.

- Evaluation: `20261007T171416Z_guard_generalist_morphology_baseline_v1_model_000000`; 256 scenarios, 64 per family; gate status PASS means the suite intentionally had no promotion gates.
- Suite-definition SHA-256: `6cf1549a86677957778ac47c7f835e81cf3f6c13a26507b2429a18c0d4738829`; resolved scenario SHA-256: `6bece67a65d848d10fe81142a92078b19c197d05c1d036f0f24443c4422b8f2b`.
- Wheel contact metrics use MuJoCo's binary `found` contact flags. The evaluator correctly reports wheel normal-force imbalance as unavailable; these measurements do not support force-based conclusions.

| Family | Target arrival | Dual-wheel contact, median | Airborne fraction, median | Single-wheel support run, p95 | Contact transitions, median | Leg asymmetry RMS, median | Leg target offset RMS, median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat precision | 38/64 (59.4%) | 0.956 | 0.000 | 0.775 s | 8 | 0.084 rad | 0.107 rad |
| Push recovery + retarget | 64/64 (100%) | 0.976 | 0.000 | 0.628 s | 7 | 0.081 rad | 0.089 rad |
| Medium target | 2/64 (3.1%) | 0.106 | 0.108 | 0.327 s | 19 | 0.805 rad | 0.729 rad |
| Long target | 0/64 (0%) | 0.000 | 0.287 | 0.170 s | 16 | 0.677 rad | 1.224 rad |

Flat precision and push-recovery behavior remain useful rollback anchors. The medium and long scenarios expose clear failure modes in the selected transfer: very low target arrival, frequent one-wheel/airborne phases, and large leg asymmetry/target excursions. This is a diagnostic finding, not a normal-mode threshold; WP-02's threshold decision remains open until pathological rollouts and good baseline behavior are compared.

The evaluator reward-term telemetry was replayed twice on the same four deterministic scenario seeds. Runs `20261007T172730Z_guard_generalist_morphology_replay_smoke_v1_model_000000` and `20261007T173034Z_guard_generalist_morphology_replay_smoke_v1_model_000000` produced identical values for all 394 episode-metric rows. The sum of timestep-scaled reward-term returns reconciled with the reported episode return; the largest absolute floating-point difference was 0.00184 over a 20-second scenario. This verifies deterministic morphology metrics and the reward-term integration units on the smoke suite.

## Guard generalist locomotion roadmap WP-03 reward alignment probe

`scripts/reward_probe.py` integrates the configured generalist reward terms over six deterministic, 11-second scripted state trajectories. These are arithmetic probes, not physics simulations or learned-policy rollouts. The full per-term artifact is `docs/experiments/reward_audit_wp03.json`.

| Probe | Final target error | Heading error | Integrated return | Relevant reward contributions |
| --- | ---: | ---: | ---: | --- |
| Successful 15 cm arrival + settle | 0.000 m | 0.000 rad | 114.679 | proximity +43.771; progress +0.784; settled balance +10.105 |
| Overshoot 15 cm target by 10 cm | 0.100 m | 0.000 rad | 110.170 | proximity +40.672; progress +0.441; settled balance +10.007 |
| Arrive with wrong heading | 0.000 m | 0.500 rad | 92.125 | heading +1.429; settled balance +1.313 |
| Arrive and keep rocking | 0.000 m | 0.000 rad | 105.688 | settled balance +1.470; dedicated body-rocking penalty −0.002 |
| Successful 0.75 m travel | 0.000 m | 0.000 rad | 116.227 | progress +12.526; proximity +36.971 |
| Successful 2.50 m travel | 0.000 m | 0.000 rad | 119.377 | progress +50.733; proximity +8.983 |

At equal duration, the 10 cm overshoot retains 96.1% of the exact-arrival total return, chiefly because `world_target_proximity` remains high and there is no one-shot completion signal. The heading term strongly distinguishes wrong heading. Long travel gains 49.949 more progress return than the short case and ends 4.698 higher in total return despite reduced proximity and settled-balance contributions. That controlled finding supports a separate completion-reward ablation; it does not establish dynamic feasibility or guarantee the same ordering on policy rollouts.

Added `target_arrival_settled_stop`, a one-shot impulse per target attempt after the target remains within 0.035 m and the evaluator-style quiet/support envelope is held for the existing 0.35 s training dwell. Its default weight is 5.0; `ASCENTO_GENERALIST_TARGET_ACHIEVEMENT_WEIGHT=0` omits the term for the control arm. The dedicated term is training-only and does not affect play/evaluation dynamics. The reward schema is now `v4`, so old checkpoints are rejected against the changed training task contract. No other reward weights changed.

## Guard generalist locomotion roadmap WP-04 permanent cohorts

Generalist training now uses a fixed environment-slot assignment for three disjoint cohorts. The default allocation is 20% precision anchor, 20% recovery/retarget anchor, and 60% generalist navigation. Environment counts use deterministic largest-remainder allocation and a fixed seeded permutation, so the 20-environment acceptance case is exactly 4/4/12 and reset order cannot change cohort identity. Setting both anchor fractions to 0.25 produces 5/5/10 in that acceptance case for the planned 50% anchor arm.

Precision anchors receive short target attempts at reset and after each settled stop. Recovery/retarget anchors have no initial target attempt; they receive the planned push at 4 s and a short retarget at 9 s, with the attempt recorded under the recovery cohort. Generalist navigation receives the staged short/medium/long target mix. Stage promotion/demotion evidence reads recovery-anchor outcomes and navigation attempts only, so precision attempts cannot promote long-distance difficulty. Neither stage changes the configured anchor allocation.

The task exports episode reliability slices and per-band attempt metrics for each named cohort, while retaining legacy `regular`/`gate` metric aliases for existing dashboards. The pre-retarget recovery hold pose is not recorded as a completed target arrival.

WP-04 acceptance: the focused curriculum suite passed 22 tests, including exact 20-environment allocation, reset persistence, disjoint episode/attempt metrics, 25/25 anchor configuration, navigation-stage stability, and target partitioning at reset. Ruff lint and formatting checks passed. The full `tests_mjlab` run reported 291 passed and the one known unrelated failure `test_quiet_balance_is_versioned_and_uses_horizon_curriculum` (expects -0.50 while the unchanged quiet-balance config is -10.0). EXP-RETENTION-01 is complete; its run record is below.

## EXP-RETENTION-01 permanent-anchor retention pilot

The selected actor was copied into the current reward-v4 task contract with `initialize-transfer`. Actor compatibility was exact (41 actor observations, six actions, matching timestep); critic, optimizer, iteration, and environment state were fresh. The transfer checkpoint SHA-256 is `0a0ba381c9866a6d9f058c5c4b187681a35c86707a3d60d81fdca8343eaf4287`.

- Matched transfer baseline: `20261007T181943Z_roadrunner_generalist_sequence_gate_v1_model_000000` — PASS, 256/256 arrivals, final-target-error p95 `0.00558308 m`.
- The first PPO update checkpoint was saved before a recovery-anchor settled-stop telemetry error stopped its launcher. After fixing the vector filtering, that checkpoint (SHA-256 `b9e96549360b7989fe21bc9b6c968ab5a59965963ec7b6daececa4e26f5caf82`) passed `20261007T183130Z_roadrunner_generalist_sequence_gate_v1_model_0` — 256/256 arrivals, final-target-error p95 `0.00594291 m`, all nine hard gates passing.
- Continuation run `430a16cbadef` resumed that checkpoint’s actor, critic, optimizer, and iteration state for 299 more iterations, for 300 optimizer updates total including the saved first update. It used fixed 20/20/60 cohorts, no teacher loss, the default arrival-settle reward weight 5.0, and checkpoint interval 50.
- Two earlier launch attempts are retained in the run ledger: one stopped before any PPO update because event parameters were not accepted by the callable; the other saved one update then failed on the now-covered recovery telemetry edge. Both issues were fixed before continuation. The held-out promotion suite remains unused.

## WP-11A deployment-observability inventory

Added `docs/reports/ascento_guard_actor_observability_inventory_2026-10-07.md` with the 41D actor channel map, likely deployment sensing/estimation paths, and privileged critic-only channels. The repository contains no hardware driver or sensor adapter. Public Research-platform documentation supports candidate paths for joint states, IMU, sensor fusion, LiDAR/depth, and GNSS/INS, and the Research FAQ says Guard shares the hardware platform; the exact Guard service configuration and policy-facing topics remain unverified.

WP-11A's inventory deliverable is complete, but this does not clear the terrain gate. Current simulated base height and wheel contact/force inputs have no verified Guard-facing contract. No observation ABI or training configuration changed.

## WP-05 frozen replay actor retention

Added a deterministic stride-10 recorder and versioned NPZ/JSON replay format. The source actor was the selected reward-v4 transfer checkpoint. Collection ran the exact 256-scenario development suite, passed its gates and collection consistency checks, and resolved to scenario hash `a7afcfe432d11d1bcc7694faf4e499ae9a62f37244b86a9a9e4b0de790892fc1` before saving the corpus. The data contains 51,200 raw 41D actor observations and deterministic teacher mean actions: 10,240 precision-anchor samples and 40,960 recovery/retarget samples. Training sampling rebalances both cohorts to 50/50.

Replay data and its sidecar are preserved at `retention/exp_retention_02_anchor_teacher_v1.npz` and `retention/exp_retention_02_anchor_teacher_v1.json`. The sidecar binds source checkpoint, actor normalizer, task/action/plant and actor-observation contracts, suite definition, and resolved scenario hashes. The loader rejects metadata/data tampering, ABI or width mismatch, missing cohort data, and non-finite arrays. PPO runs the deterministic-mean MSE update after ordinary PPO and optimizer state updates only actor parameters; zero coefficient returns before touching the model or optimizer. It logs the weighted auxiliary loss, coefficient, balanced sample count, actor/teacher action RMS, actor gradient norm, and policy-standard-deviation parameter delta.

WP-05 focused replay and scenario tests passed (18 total), and Ruff passed for the replay, evaluator hook, PPO, runner, collector, and test files. EXP-RETENTION-02 used fixed `lambda_ref=0.1`, replay batch size 256, the same 20/20/60 cohorts and seed, with checkpoints screened on the same development suite. The run completed 300 updates, and all seven planned checkpoints were evaluated.

### EXP-RETENTION-02 completed results

Run `cb21b3ebbc21` completed with exit code 0. Replay weight was 0.1; update batches used 128 precision samples and 128 recovery/retarget samples. The replay action RMS was 0.0078 early in training, 0.0241 at update 50, 0.0715 at update 100, 0.1354 at update 150, and 0.1875 at the final checkpoint. The actor-only update logged policy-standard-deviation parameter delta 0.0 throughout.

| Update | Gate | Arrivals | Final target-error p95 | Paired target-error IQM delta vs transfer (95% CI) | Paired IQM delta vs EXP-RETENTION-01 (95% CI) | Replay action RMS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | PASS | 256/256 | 5.98 mm | +0.19 mm (+0.15, +0.23) | +0.02 mm (+0.01, +0.02) | 0.0019 |
| 50 | PASS | 256/256 | 7.40 mm | +1.25 mm (+1.16, +1.34) | +0.17 mm (+0.07, +0.28) | 0.0241 |
| 100 | PASS | 256/256 | 15.95 mm | +9.91 mm (+9.78, +10.05) | +1.24 mm (+1.07, +1.41) | 0.0715 |
| 150 | PASS | 256/256 | 55.90 mm | +49.65 mm (+49.46, +49.82) | +7.13 mm (+6.96, +7.32) | 0.1354 |
| 200 | FAIL | 256/256 | 119.69 mm | +112.22 mm (+111.93, +112.48) | +18.52 mm (+18.36, +18.70) | 0.1896 |
| 250 | FAIL | 256/256 | 156.43 mm | +148.15 mm (+147.82, +148.48) | +29.19 mm (+29.03, +29.35) | 0.1928 |
| 299 | FAIL | 256/256 | 162.36 mm | +154.23 mm (+153.95, +154.55) | +28.41 mm (+28.25, +28.58) | 0.1875 |

The fixed replay arm delayed its first formal hard-gate failure from update 150 to update 200, but its target-error p95 at update 150 was already above the project's practical 50 mm precision target. More importantly, paired target-error IQM was worse than EXP-RETENTION-01 at every matched update; the final paired difference was +28.4 mm with a 95% interval excluding zero. Against the historical reward-BC checkpoint (`model_99`, BC weight 2.0, frozen-transfer normalizer), replay was better at updates 0 and 50, slightly worse at 100, and materially worse thereafter. All checkpoints still arrived on all 256 scenarios.

Conclusion: EXP-RETENTION-02 does not satisfy the planned better-tradeoff criterion. Replay-set behavior itself drifted substantially, so the condition for EXP-RETENTION-03 (good replay retention with only on-policy anchor drift) is not met; do not add live rollout masking on this evidence. The selected transfer remains the rollback floor. Detailed hashes, per-checkpoint comparisons, and confidence intervals are in `docs/experiments/exp_retention_02_results.json`; the held-out promotion suite was not used.

### EXP-RETENTION-01 completed results

The 300-update run 430a16cbadef completed successfully as a trainer run. It used the 20/20/60 permanent cohorts, ordinary PPO, no teacher loss, and the WP-03 target-arrival bonus. Every checkpoint was screened against the same 256 resolved development scenarios (a7afcfe4...); the held-out promotion suite was not used.

| Checkpoint index | Gate | Arrivals | Final target-error p95 | Paired target-error IQM delta vs transfer (95% CI) |
| ---: | --- | ---: | ---: | ---: |
| 0 (first update) | PASS | 256/256 | 5.94 mm | +0.17 mm (+0.14, +0.21) |
| 50 | PASS | 256/256 | 6.94 mm | +1.05 mm (+0.94, +1.15) |
| 100 | PASS | 256/256 | 14.23 mm | +8.49 mm (+8.34, +8.61) |
| 150 | FAIL | 256/256 | 49.33 mm | +42.26 mm (+41.96, +42.54) |
| 200 | FAIL | 256/256 | 101.64 mm | +93.61 mm (+93.20, +94.02) |
| 250 | FAIL | 256/256 | 127.68 mm | +118.96 mm (+118.49, +119.43) |
| 298 (final) | FAIL | 256/256 | 134.19 mm | +125.94 mm (+125.48, +126.38) |

The paired intervals show increasing precision loss despite fixed anchor slots. The index-150 checkpoint failed other hard gates even though its target-error p95 was just below the 50 mm practical ceiling; indices 200 and later exceeded the suite's 100 mm formal limit as well. All checkpoints still recorded target arrival on all scenarios, so arrival alone concealed a substantial quality regression.

Conclusion: EXP-RETENTION-01 does not pass retention. The selected actor remains the transfer quality floor. Proceed to WP-05 / EXP-RETENTION-02 with a frozen anchor observation replay and actor-only mean-action MSE. Detailed hashes and paired intervals are in docs/experiments/exp_retention_01_results.json. The separate WP-11A inventory remains complete with terrain readiness false.


## WP-06 semantic observation normalization

Added `SemanticNormalizationMLP` and a semantic normalizer contract for the existing 41D actor and 50D critic inputs. Running normalization covers the 22 continuous actor proprioceptive channels and the critic's nine additional privileged state channels. World-target XY uses fixed 1 m scaling, heading uses fixed pi-radian scaling, and contact booleans, pre-scaled contact forces, normalized actuator effort, and previous normalized actions retain identity/fixed scaling. The task observation and action ABI remain unchanged.

The migration utility transforms actor and critic first-layer weights and biases analytically using the pinned RSL-RL normalization formula `(x - mean) / (std + 0.01)`. The serialized checkpoint contract selects the matching model in training, evaluation, and viewer loading. Migration is restricted to an iteration-zero checkpoint with empty optimizer state so optimizer moments cannot be left in the old coordinates.

### EXP-NORM-00 complete

The selected reward-v4 transfer checkpoint was migrated without learning. On all 51,200 frozen replay observations, deterministic actor outputs matched with maximum absolute error `4.77e-7` and RMS error `3.89e-8`. Both source and migrated checkpoints passed the same 256-scenario development suite with the same resolved-scenario hash; the paired comparator matched 256 scenarios and the largest absolute paired IQM metric delta was `5.41e-6`. Final-target-error p95 was `5.57 mm` for the source and `5.58 mm` for the migrated checkpoint. The held-out promotion suite was not used. Full hashes and evaluation IDs are in `docs/experiments/exp_norm_00_results.json`.

### EXP-NORM-01 completed results

Managed run `c27772ee729b` completed all 300 updates from the equivalence-verified migrated checkpoint with 512 environments and seed `20261007`. Semantic normalization was the only planned change; the 20/20/60 cohorts, reward-v4 task, optimizer profile, and six-action ABI were fixed. Replay retention and reward-BC losses were disabled.


| Update | Gate | Target-error p95 | Paired target-error IQM delta vs transfer (95% CI) |
| ---: | --- | ---: | ---: |
| 0 | PASS | 8.81 mm | +2.99 mm (+2.83, +3.13) |
| 50 | PASS | 10.03 mm | +4.44 mm (+4.28, +4.58) |
| 100 | PASS | 21.42 mm | +16.01 mm (+15.84, +16.16) |
| 150 | PASS | 43.07 mm | +37.20 mm (+37.02, +37.39) |
| 200 | PASS | 79.02 mm | +72.88 mm (+72.71, +73.04) |
| 250 | PASS | 89.94 mm | not available for all 256 rows |
| 299 | FAIL | 113.86 mm | not available for all 256 rows |

All seven checkpoints used the same 256 resolved scenarios (SHA-256 `a7afcfe4...892fc1`) and the same development-suite definition. The held-out promotion suite was not used. Update 150 is the last checkpoint that passes both the formal gates and the 0.05 m practical p95 limit. Against the matched update-150 controls, its paired target-error IQM is lower by 5.30 mm versus EXP-RETENTION-01 (95% CI [-5.62, -4.98] mm) and 12.45 mm versus EXP-RETENTION-02 ([-12.60, -12.30] mm). Update 200 also beats the matched retention controls, but its 79.02 mm p95 misses the practical limit. At update 299 the checkpoint fails survival, recovery, final target error, and stationary body-rocking gates. The comparison tool omits final-target-error paired intervals where the metric is missing for any scenario; those late checkpoint results remain in the full evaluation artifacts.

The semantic normalizer improves the late-training tradeoff relative to the matched control arms, but update 150 still has a statistically clear +37.20 mm target-error IQM regression versus the selected transfer. Do not promote the trained policy; the transfer remains the quality floor. EXP-NORM-01 is complete as a diagnostic/control-tradeoff result. Next on the critical path is WP-07 and its mode-zero transfer equivalence check; this is evaluation-only. WP-08 terrain work remains gated by retention and the open Guard sensor/contact/height contracts.

Full checkpoint IDs, SHA-256 values, pairwise comparisons, and the promotion decision are in `docs/experiments/exp_norm_01_results.json` and `evaluations/comparisons/exp_norm_01/`.

## WP-07 / EXP-CONTEXT-01 — mode-zero context transfer passed

The generalist task now carries one explicit Boolean-float `obstacle_mode` at actor index 41 and critic index 41. It defaults to zero, remains identity-scaled in semantic-normalization v2, and can be set to 0 or 1 from the evaluation and viewer command lines. The six-action ABI is unchanged. The task contract fingerprints the new 42D actor and 51D critic and rejects the former 41D checkpoint contract.

The iteration-zero context migration started from the already equivalence-verified semantic checkpoint. Its mode column was initialized to zero and the checkpoint was evaluated without PPO updates. On the frozen 51,200-observation corpus, all six outputs matched exactly (maximum and RMS deltas both zero). The migrated policy passed all gates on the same 256 development scenarios as the source. Paired output metrics had a maximum absolute IQM delta of `1.91e-6`; target-error p95 was `5.57 mm` versus `5.58 mm` for the source. The held-out promotion suite was not used.

WP-07 and EXP-CONTEXT-01 are complete. Continue with the WP-07B reward-freedom audit. Terrain remains gated by that audit and the unresolved Guard-specific height/contact/force deployment contracts in WP-11A. Full results and hashes are in `docs/experiments/exp_context_01_results.json`.

## WP-07B reward-freedom control and terrain gate

The reward audit quantified symmetry, leg-hold, fixed-height tracking, and the fixed-height factor inside settled balance across the five required probe classes (six static cases). Scripted useful postures were directly penalized: the asymmetric split-wheel case lost 3.265 reward/s from symmetry and leg hold, while the low-obstacle leg excursion lost 2.937 reward/s from those terms. A 10 cm height change lost 0.790 reward/s from height tracking and 0.982 reward/s from settled balance's fixed-height factor.

WP-07B now conditions only these demonstrated conflicting terms on `obstacle_mode=1`: the symmetry and leg-hold penalties are removed, and height objectives permit a ±0.10 m band. Mode 0 leaves prior rewards unchanged and retains a 3.252 reward/s penalty for the scripted unnecessary pseudo-step. This is a new reward/task contract (`f27347eee8a2fcb7bae2af15a1996d13e1d0afcb815c11d48293fc5d3159e16c`). The evidence is static reward arithmetic and does not establish dynamic feasibility or learned skill.

The zero-update migrated checkpoint passed the same 256-scenario mode-zero development suite: 256/256 arrivals and 5.569 mm target-error p95. The paired target-error IQM delta versus the prior mode-zero control was 0.00012 mm (95% CI −0.00047 to 0.00071 mm). Evaluation and comparison artifacts are under `evaluations/exp_reward_freedom_01/` and `evaluations/comparisons/exp_reward_freedom_01_mode0.json`.

The planned 150–250 update reward-freedom training ablation remains deferred. WP-11A found that current actor base height, wheel contact flags, and wheel contact-force norms are simulator-only as defined or have no verified Guard deployment path. WP-08 prohibits terrain adaptation from relying on those channels. Define a deployment-feasible observation contract (or replace/move those inputs) before terrain/obstacle training; do not read the static probe or mode-zero control as evidence of mode-one learning.

### Final verification status

The full `tests_mjlab` run finished with 315 passed and one known unrelated failure: `tests_mjlab/test_task_config.py::test_quiet_balance_is_versioned_and_uses_horizon_curriculum` expects the unchanged quiet-balance reward weight to be −0.50 while the config sets −10.0. The focused curriculum/checkpoint/viewer regression suite passed all 31 tests. Ruff lint, formatting for all changed Python files, JSON parsing, and `git diff --check` passed. This does not change the terrain training gate described above.

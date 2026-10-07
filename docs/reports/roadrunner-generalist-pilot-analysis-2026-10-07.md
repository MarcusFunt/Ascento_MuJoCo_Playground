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

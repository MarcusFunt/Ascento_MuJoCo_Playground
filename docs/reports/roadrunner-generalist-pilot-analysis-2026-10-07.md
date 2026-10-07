# Roadrunner-inspired curriculum pilot analysis

**Date:** 2026-10-07
**Practical target precision:** final-target-error p95 at or below **0.05 m**, per the project owner's clarification. The fixed suite's formal hard gate remains **0.10 m**.
**Decision:** the frozen-reference BC checkpoint is a usable experimental candidate and meets both target-error limits, but it is **not a proven upgrade** over the actor-only transfer. Keep the transfer policy as the selected policy.

## Executive summary

The earlier fine-tuning pilots show that higher training reward and a larger gate-like share do not reliably preserve the transfer policy's target-and-stop behavior. A lower-learning-rate/tighter-KL arm also regressed, so optimizer size alone does not explain the loss. The implemented task-slice telemetry confirms that the success-gated long-goal mix remained at stage 0: by the BC continuation's last complete telemetry, gate recovery's Wilson lower bound was 0 and short-arrival's was about 0.20, below the stage thresholds of 0.85 and 0.50.

A frozen-reference behavior-cloning (BC) term was added as a separate, opt-in stabilization arm. Its checkpoint at iteration 100 passed all nine suite gates, arrived at the target in 256/256 scenarios, and had final-target-error p95 **0.01818 m**, comfortably within the requested 0.05 m tolerance. Paired evaluation showed a statistically clear target-error regression against the transfer baseline (IQM +0.01196 m, 95% CI +0.01177 to +0.01217 m), offset by better heading and stationary tilt, with small in-gate regressions in action smoothness and rocking. It was **PASS / NOT_PROVEN_BETTER**.

The continuation's first attempt stopped at iteration 158 with CUDA 999 surfaced inside an unrelated leg-pose reward and no new checkpoint. An exact same-settings retry completed through iteration 199. Its checkpoint passed all nine formal gates, but target-error p95 rose to **0.08746 m**, outside the requested 0.05 m tolerance, and it was significantly worse than both iteration 100 and the transfer on target error, post-target heading, stationary tilt, and body rocking. Neither checkpoint is promoted.

## Evaluation setup

- Fixed suite: `roadrunner_generalist_sequence_gate_v1`, 256 resolved scenarios, scenario SHA-256 `a7afcfe432d11d1bcc7694faf4e499ae9a62f37244b86a9a9e4b0de790892fc1`.
- Baseline: signed actor-only transfer `transfers/ascento_generalist_locomotion_flat/roadrunner_generalist_signed/model_000000.pt`; evaluation `20261007T085540Z_roadrunner_generalist_sequence_gate_v1_model_000000`; all nine gates pass, target-error p95 0.00557 m.
- BC candidate: run `c93a9c51dff2`, checkpoint `model_100.pt`, SHA-256 `c356bb95e2d17c12f809dd370ccfb2393643b7fddba278cbd5d6743ff5f090ad`; evaluation `20261007T104927Z_roadrunner_generalist_sequence_gate_v1_model_100`. The candidate and baseline used the same suite/scenario identity and passed report consistency checks.
- Training configuration: 512 environments, seed 20261007, PPO defaults, 25% final gate-like fraction, frozen teacher copied from the same actor-only transfer, BC weight 2.0. Training and evaluation used the same signed task contract and BC environment settings.

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

## Root-cause findings and limits

1. **Curriculum advancement did not follow demonstrated recovery quality.** Training now reports task slices, goal bands, curriculum progress, and Wilson bounds. In the BC continuation, progress was about 0.15, gate-like fraction about 0.1225, gate-recovery LCB 0, and short-arrival LCB about 0.20. The success-gated goal-mix stage remained 0; the sampled short/medium/long fractions were approximately 0.65/0.25/0.10. The intended curriculum signal exists, but the policy was not meeting its advancement thresholds.
2. **More gate-like episodes were insufficient.** Both 25% and 40% arms first failed formal gates at checkpoint 200; target-error p95 was worse in the 40% arm (0.0697 m vs. 0.0529 m). Episode frequency alone does not teach the exact post-push retarget, arrival, and quiet-stop sequence.
3. **Reducing PPO step size was insufficient.** The conservative arm still failed by checkpoint 200, so optimizer drift is not the only likely cause.
4. **The BC term is not yet calibrated as a reliable trust region.** It yielded an iteration-100 checkpoint inside the 5 cm target, but target-error IQM was already worse than baseline. During the first continuation attempt the per-step BC reward reached about −0.0105, and on the successful retry it reached about −0.0751 near iteration 199. The iteration-199 policy's target p95 and other paired metrics were substantially worse. This indicates increasing divergence from teacher actions; it does not isolate whether weight, student-normalizer drift, or states outside the teacher's distribution caused it.
5. **Normalizer drift remains a plausible contributor, not an isolated cause.** Prior checkpoint inspection found target-error observation-normalizer scales about 2.3–2.4 times their initial values at later checkpoints. There is not yet a persisted per-checkpoint normalizer summary in managed telemetry, so this result needs a reproducible report and a separate frozen-normalizer ablation.
6. **There is a separate GPU reliability issue.** Earlier BC launch attempts exposed a CPU-only PyTorch environment and an asynchronous Warp/CUDA 999 error. Adding a targeted CUDA synchronization let the parent BC run reach iteration 196 and its iteration-100 evaluation pass. One continuation failed at iteration 158 inside `leg_pose_hold_penalty`; the exact retry completed through iteration 199 and evaluated with no CUDA error. No PPO update was invalid in either continuation, so the failure is runtime, not policy quality. A prior evaluation also failed while sharing the GPU with a trainer; the isolated reevaluation passed.

The experiments support curriculum/reward mismatch and observation-normalizer drift as hypotheses. They do not prove either is the sole cause.

## Recommended next experiments

1. Keep the signed actor-only transfer as the deployed/default policy and fixed-suite baseline.
2. Preserve the 5 cm practical final-target-error threshold and 10 cm formal suite gate. Screen every checkpoint on the same 256 scenarios; report both paired deltas and absolute target p95.
3. Add a reproducible checkpoint report for actor observation-normalizer mean/scale, then change only normalization: compare adaptive statistics with preserved transfer statistics while keeping optimizer, mixture, and seed fixed.
4. Make gate-like precision tasks present at their intended final fraction from the start in one ablation, while leaving the long-goal curriculum's success-based stage rules and PPO settings fixed. Compare against the current time-ramped gate fraction.
5. Add a short-task reward ablation only after checking per-term returns for episodes that arrive within 0.035 m and settle with the requested heading. If those outcomes are under-rewarded, add an explicit arrival/settle term and test it alone.
6. Recalibrate reference regularization in a separate sweep (for example, a lower BC weight and a slow schedule tied to gate-like episodes). First demonstrate that it preserves target p95 within 0.05 m and does not worsen guarded rocking/action metrics; do not infer success from return.
7. For each arm, save/screen at 50–100 iteration intervals initially, stop on the first sustained gate failure, and do not start a second seed until a candidate passes all gates and shows an acceptable paired tradeoff.
8. Run only one CUDA workload at a time. Keep `--extra cu128` active, verify the resolved PyTorch CUDA device before launch, record the first CUDA exception separately from cleanup errors, and use `CUDA_LAUNCH_BLOCKING=1` only for bounded diagnosis because it cut throughput to roughly one quarter.

## Verification

Focused tests for the regularizer, curriculum, and reward contract passed (21 tests); the CUDA regularizer test passed after the synchronization change (4 tests); Ruff passed on changed Python files. The full suite reported 406 passed and one existing, unrelated mismatch: `tests_mjlab/test_task_config.py::test_quiet_balance_is_versioned_and_uses_horizon_curriculum` expects reward weight −0.50 while the unchanged quiet-balance configuration sets −10.0.

# GOAL-SETTLE-v2: matched 50-update controlled pilot (2026-10-12)

## Decision

**Do not promote GOAL-SETTLE-v2 or scale the treatment to a multi-hour run yet.** Both arms retained stationary balancing and straight-line stopping on a four-scenario matched physical diagnostic. Neither reached the offset waypoint. The treatment improved straight final position by roughly 4 mm while slightly worsening the 25 cm final position and the offset-waypoint error. Four deterministic scenarios with one seed are not a population-level test; the result provides **no convincing evidence** of improved navigation.

## Intervention implemented and validated

Merged PR #126 introduces the opt-in task `Ascento-Generalist-Locomotion-Goal-Settle-V2-Flat`. The original task remains available. Treatment uses more near-goal signed progress, delays braking near the goal, distance-conditions heading and settled-balance shaping, and aligns an 8-point one-shot arrival reward with the event's 10 cm / 0.20 rad / 0.50 s completion envelope. This is a combined reward and event-contract change, not a single-term causal ablation. Merged PR #127 ensures invalid managed resume contracts are recorded as errors and documents exact stage-0 transfer construction.

Preflight: **90 focused unit, reward and evaluator tests passed**; Ruff passed. In real mjlab steps, weighted reward-term rates sum to the delivered reward within 1e-8. A scripted 12-second objective-ranking test prefers successful settling to parking 11.5 cm away. These tests validate calculation and configuration, **not learned physical success**.

## Paired initialization and train setup

Both actor weights and actor observation normalizers were copied from the preserved `reward_alignment_reward_fix/model_000000.pt` actor (source SHA-256 prefix `c79062d4a9b7e93f`). Both fresh initialization checkpoints have **exactly identical actor and critic tensor values**, empty optimizer states and iteration 0. Their signed task-contract hashes are different as required. The comparison initializer paths are:

- Control: `transfers/ascento_generalist_locomotion_flat/goal_settle_comparison_stage0/model_000000.pt`
- Treatment: `transfers/ascento_generalist_goal_settle_v2_flat/goal_settle_comparison_stage0/model_000000.pt`

Both used 512 environments, 24 rollout steps/update, seed `20261012`, fixed goal-mix stage 0, 50 updates, learning rate 0.0001, gamma 0.998 and save interval 10. Runs launched **sequentially** through the clean-main `scripts/ascento-gpu` guard. Each exited successfully with code 0 and saved models 0,10,20,30,40,49.

- Control managed run ID: `994294ebafdf` ? `logs/rsl_rl/20261012_210822_goal-settle-v2-matched-control-50_994294eb`
- Treatment managed run ID: `1126a8b9d658` ? `logs/rsl_rl/20261012_211107_goal-settle-v2-matched-treatment-50_1126a8b9`
- Frozen experiment pairing manifest: `/tmp/ascento_goal_settle_matched_comparison_manifest_20261012.json`

The training log's final mean reward is **23.21 control vs 28.50 treatment**, but because reward functions differ those values are **not directly comparable**. Training-window short-attempt counts were 25/98 control (3.5 cm threshold) and 59/120 treatment (10 cm threshold), also **not apples-to-apples**. Both reported stage 0, with no evidence supporting curriculum promotion.

## Paired immutable physical diagnostic

Both final policies were evaluated in the **original generalist play environment**, with unchanged physical completion rules; treatment `model_49.pt` was copied actor-only into a v1 evaluation checkpoint and verified tensor-identical. The evaluator required `ASCENTO_GENERALIST_FIXED_GOAL_MIX_STAGE=0` to match checkpoint contracts. Two early attempts without this environment variable were correctly marked INVALID and excluded from analysis.

- Suite: `/tmp/ascento_goal_settle_paired_smoke_20261012.toml`
- Suite hash: `feb19228149f4b58945bf356e41f6cf3594840e44b46f65f2d05636dc6e1e475`
- Resolved scenario hash: `81a63a5b9f26ae478a52e08da4a43a8e8beca16bcbeba6f38060e306c009a503`
- Control evaluation: `evaluations/20261012T191743Z_goal_settle_paired_smoke_20261012_model_49`
- Treatment evaluation: `evaluations/20261012T192118Z_goal_settle_paired_smoke_20261012_model_000000`
- Both evaluations: **consistency PASS, checkpoint contract current, hard diagnostic arrival gates PASS**.

| Scenario | Control final error | Treatment final error | Arrived control/treatment | Settled control/treatment |
|---|---:|---:|:---:|:---:|
| hold_3s | 0.08 cm | 0.25 cm | 1 / 1 | 1 / 1 |
| precision_025m | 1.13 cm | 1.23 cm | 1 / 1 | 1 / 1 |
| straight_050m | 2.12 cm | 1.71 cm | 1 / 1 | 1 / 1 |
| waypoint_056m | 28.48 cm | 28.75 cm | 0 / 0 | 0 / 0 |

The 25 cm and 50 cm arrivals satisfy the same evaluator thresholds in both arms. Both miss the 56 cm offset waypoint by approximately 28?29 cm. This 4-scenario suite is an **early smoke regression screen**, not the official multi-seed development gate or the 320-scenario reward-alignment benchmark. Its hard PASS applies only to the precision and straight arrival gates defined in this small suite.

## Next gate, before any longer PPO training

1. Construct a **paired, fixed-seed near-goal stress suite**: start at 11?15 cm position error and explicitly vary heading error, planar speed, yaw, and push/recovery conditions. Keep 10 cm evaluator completion unchanged.
2. Evaluate control vs v2 model 0/10/20/30/40/49 on identical scenario hashes. Record arrival, final position, settled dwell, heading, survival and action-frequency metrics. A larger sample and at least two additional random seeds are required before claiming repeatable improvement.
3. If the near-goal stress test does not favor v2, isolate one reward/event intervention at a time rather than simultaneously modifying braking, heading, progress and completion bonus.
4. Only after retaining original short-goal reliability and improving difficult waypoint completion, run the official hard sequence gate and the complete reward-alignment suite. **Do not promote based on training reward or the four-case smoke alone.**

## Provenance

Source commit at launch: `4bc930c` (main, clean). Experiment source code merged in PR #126 and launcher fix in PR #127. This report is observational; it does not modify any learned policy, trained checkpoint or immutable evaluation.

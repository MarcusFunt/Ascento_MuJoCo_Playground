# Goal-Settle V6: temporary training-arrival envelope and Rerun diagnosis

Date: 2026-10-13. Status: **isolated experimental implementation, not a promoted policy**.
Do not stop or alter V5 managed run `071a5acc4409` while it is learning.

## Why this experiment

The V5 30-point training completion impulse requires **50 consecutive 10 ms samples** satisfying all of: distance <=0.10 m; heading error <=0.20 rad; tilt <=0.16 rad; planar speed <=0.10 m/s; XY angular speed <=0.25 rad/s; yaw speed <=0.25 rad/s; height error <=0.05 m; **both wheels supported**. One failed sample resets the dwell timer. The real target-retarget event has a matched envelope. This is an intentionally difficult near-goal objective and can starve PPO of completion credit.

Loosening *only* the evaluation suite's settled-arrival gate would not change PPO at all. Conversely, relaxing the target-retarget event but not the reward impulse would misalign training. V6 modifies the event and reward **together** and preserves the original V5 task and independent strict evaluator.

## Paired Rerun evidence (initial single-seed 16 s screen)

We loaded immutable checkpoints 16,300 and 19,199 into the **same V5 training-mode** MuJoCo task, with a 3-environment minimum cohort allocation and a single recorded observation stream (environment 0). Seed/take was zero, deterministic inference, physics and reward contract V5, same initial scenario. Both captures have 1,600 frames and exact reward summation parity (better than 2e-8 absolute).

Locations under the canonical checkout:

- `captures/goal-settle-v5-model16300/take_000.npz` and `take_000.rrd`
- `captures/goal-settle-v5-model19199/take_000.npz` and `take_000.rrd`

The two 16-second captures are small **diagnostics, not benchmark statistics**. They use training-mode rewards for faithful reward inspection, so they are not the strict play/evaluation environment.

| Observed signal | V5 model 16300 | V5 model 19199 |
|---|---:|---:|
| Frames <=10 cm of target | 1548/1600 | 1564/1600 |
| Near frames failing <=0.20 rad heading | 1289 | 1500 |
| Near frames lacking both wheel contacts | 1143 | 12 |
| Near frames exceeding 0.16 rad tilt | 3 | 0 |
| Near frames exceeding 0.10 m/s speed | 820 | 12 |
| Actual one-time 30-point completion payouts | 1 | 0 |
| Mean planar root speed | ~0.12 m/s | ~0.01 m/s |

The sets of failed conditions overlap and the capture does not contain per-frame angular velocity; these counts alone **cannot** reconstruct the complete settled-gate truth value. Still, the collapsed model mostly stays near the goal but has a very large heading miss. This is more diagnostic than simply comparing mean reward. The good model earned one completion but exhibited extensive contact intermittency while getting there.

Do not conclude that changing the gate will fix steering or eliminate catastrophic forgetting. Test whether it increases completed *physically settled* tasks, rather than merely widening an arbitrarily easy label.

## V6 opt-in training intervention

| Parameter | V5 training | V6 training |
|---|---:|---:|
| Arrival distance | 0.10 m | **0.12 m** |
| Heading tolerance | 0.20 rad | **0.28 rad** |
| Tilt | 0.16 rad | **0.20 rad** |
| Uninterrupted dwell | 0.50 s | **0.25 s** |
| Planar speed / angular rates / height / both contacts | unchanged | unchanged |
| One-shot bonus magnitude | 30 | 30 |
| Bounded settled-quality max total | 6 | 6 |
| Pending-goal cost and fall penalty | unchanged | unchanged |
| Strict evaluator | unchanged | unchanged |

V6 smooth quality envelope is slightly wider (max goal distance 0.15 m, heading 0.45 rad, tilt 0.28 rad, planar speed 0.14 m/s) so the agent receives bounded near-goal feedback. No far-field passive reward is introduced and no unlimited circle-occupancy bonus is added. All semantics are preserved on V5.

**Important:** This changes reward topology and target-reset semantics, so a V5 full PPO resume is invalid. Use an actor-only V5-to-V6 initialization, fresh critic and optimizer, signed new task and new experiment directory.

## Suggested controlled next experiment (after V5 and guard finish)

1. Preserve the V5 model 16,300 SHA-256, current rollback continuation and RRD replays; do not promote model 19,199.
2. Prepare **paired control V5 vs treatment V6** from the *same* V5 actor, each with fresh critic/optimizer/iteration and copied actor normalization; matching seed, 512 environments, obstacle off, fractions 0.15 precision / 0.05 recovery / 0.80 navigation, heading quarter turns 0.05 and automatic curriculum. RSL-RL goal-settle V5-compatible actor loader should be used via the existing `initialize_transfer` helper. Never just change the task name on a V5 checkpoint and resume optimizer.
3. Run a small 100–250 PPO update pilot of each, sequentially, with a conservative optimizer profile only if **both arms** use the same profile. Record completion reward frequency, per-target completion, heading error at settle, time-to-settle, falls, speed and stage. Keep a few checkpoints for retrospective replays.
4. For both arms, produce matched NPZ+RRD captures of identical physical scenarios. Judge using the **unchanged strict V5 play evaluator** (or evaluation actor transfers back into V5 with verified actor SHA-256 and task-contract compatibility).
5. Promote V6 only if strict settled completions rise without loss of heading, long-goal arrival, upright survival and recovery. If relaxation mainly produces coarse stops with poor heading, revert and address heading policy/observations explicitly.

No V6 training was started here; the active V5 GPU training and its overnight regression monitor remain untouched.

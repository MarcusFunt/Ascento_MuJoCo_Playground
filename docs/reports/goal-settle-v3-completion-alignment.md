# GOAL-SETTLE-v3: completion-first reward alignment

**Status:** implemented as a separate task; **not behaviorally validated** until fixed-seed PPO rollouts and physical evaluations pass. This change does **not** modify the active GOAL-SETTLE-v2 training run, existing checkpoints, task contracts, observations, actuators, or evaluation thresholds.

## Reproduced failure and diagnosis

In the fresh-from-scratch GOAL-SETTLE-v2 rollout near update 200, the live monitor repeatedly sampled a robot **~0.9–1.3 cm from its waypoint**, traveling/oscillating at **~0.035–0.183 m/s**. Both-wheel contact was intermittent, the one-time completion bonus remained **zero**, and positive step rewards persisted (~0.006–0.074). The measured 11-sample mean progress reward was approximately zero. This is *not* an explicit proximity reward; it is **ongoing alive/upright/height/heading credit while the attempt stays unfinished**. The original v2 completion requires 0.5 continuous seconds within 10 cm, heading error <=0.2 rad, support on both wheels, and quiet posture/motion.

Simply multiplying velocity progress is insufficient: at a stationary or back-and-forth target-relative position, signed progress averages zero. Adding another uncapped near-goal shaping reward would reintroduce the exploit.

## Opt-in v3 task and objective

Task ID: `Ascento-Generalist-Locomotion-Goal-Settle-V3-Flat`. PPO experiment root: `ascento_generalist_goal_settle_v3_flat`.

Relative to v2 (per-second reward-manager weights except impulse bonuses):

| Term | v2 | v3 | Purpose |
|---|---:|---:|---|
| Alive | 1.0 | 0.15 | Remove passive income while retaining termination cost |
| Upright | 2.0 | 0.45 | Retain balance incentive |
| Height | 1.0 | 0.20 | Retain posture |
| Final heading | 1.0 | 0.25 | Prevent heading from dominating translation |
| Yaw-rate tracking | 0.30 | 0.10 | Reduce passive credit |
| Settled balance | 1.0 | 0.25 | Favor safe settling without continual high income |
| Signed target progress | 10.0 | **14.0** | Travel needs genuine closing velocity; retreat remains negative |
| Speed penalty near goal | -2.0 | **-3.0** | Prefer braking to oscillating through the waypoint |
| Settled-stop completion (one-shot) | 8.0 | **30.0** | Reward genuine full-dwell completion |
| Bounded settled quality | absent | **6.0 total maximum per attempt** | Dense feedback for good near-goal position/heading, two-wheel support, quiet velocity, tilt, angular motion and height |
| Pending-goal time cost | absent | **-1.70/s**, rising to **-3.00/s** by 8 s | Make indefinite noncompletion unprofitable; clock resets on target-attempt change, not entering goal circle |
| Failure termination | absent | **-200 once** (not at timeout) | Prevent deliberate falls/early reset as a loophole around increasing time cost |

The **maximum positive passive rate** from the six listed balance/heading terms is only **1.40/s**, below the **minimum unfinished-attempt cost** of **1.70/s**. All other unlisted safety and effort penalties remain negative. A one-time non-timeout failure penalty of 200 units avoids the unintended alternative of intentionally falling to escape the increasing time cost; this must be assessed for excessive risk aversion in learned behavior. The bounded settled-quality credit cannot be collected indefinitely. Therefore the parked-in-circle strategy no longer has a positive indefinite reward rate, even when all passive rewards are ideal.

### Technical implementation

`GeneralistPendingGoalTimeCost` is a stateful reward-manager term that tracks elapsed time per environment and target attempt. It detects new `attempt_id` values, including when the next attempt has the same target coordinates, as well as spatial target changes and environment resets. An inactive recovery anchor with `attempt_id < 0` is not charged before its target is issued. A goal can only reset the attempt clock through the regular event system, which requires successful settled dwell for retargeting. The time cost ramps linearly over 8 seconds and is bounded; nothing grows without limit.

The 42-dimensional actor observation contract, six-action controller and safety terms, goal distance bands, fixed cohort assignments, arrival radius (0.10m), heading tolerance (0.20rad), and dwell (0.50s) are unchanged. Evaluator completion thresholds are not weakened. Play/evaluation dynamics remain unchanged; only the new training task applies the revised reward topology.

## Verification and experiment gate

New tests: `tests_mjlab/test_goal_settle_v3.py`. They check registration and contract isolation, passive reward upper bound, escalating cost with same-coordinate retargets, inactive recovery attempts, reset, cost parameter validation, capped settled-quality credit, support-dependent completion and once-only payment, signed progress, and reward-manager sum parity on real CPU mjlab steps.

**Code/unit passing is not navigation success**. Before choosing v3 for longer training:

1. Run the focused tests and lint; exercise a short fresh-initialized PPO pilot on v3 and a **matched v2 control**, same random seed, target mix, optimizer, env count and evaluation suite. Do not resume a v2 optimizer/checkpoint directly into v3; initialize a separately signed v3 checkpoint if actor transfer is desired.
2. Evaluate fixed physical trajectories: goal 11–15 cm away; deliberate fall/early-reset attempts; 25 cm straight; 50 cm straight; 50 cm lateral offset; initial position already inside the 10 cm circle, alternating single-wheel support and yaw-rate perturbations. Compare **actual settled completions**, dwell time, target error, straight/lateral travel, falls, and wheel contacts, not mean training reward.
3. Audit reward-component time series: `pending_goal_time_cost`, `target_settled_quality`, `target_arrival_settled_stop`, `world_target_progress`, and `generalist_curriculum_goal_mix_stage`. The success criterion is **more completed target attempts, better settled behavior, and no collapse of balance**, not a larger scalar return.
4. **Rollback** if v3 parks near the circle without completing, falls frequently, regresses short-goal evaluation, or fails longer waypoint travel. Do not promote merely because the summed return changed (reward scales are intentionally different).

To start a fresh run *when the GPU is available* (the previous v2 overnight run may still be using it):

```bash
scripts/ascento-gpu ascento run start --json \
  --task Ascento-Generalist-Locomotion-Goal-Settle-V3-Flat \
  --display-name goal-settle-v3-fresh-pilot \
  --envs 512 --iterations 200 --seed 20261013 -- \
  --agent.save-interval 50
```

No GPU experiment is claimed here: remote access to Marcus_Computer was unavailable while authoring this change.

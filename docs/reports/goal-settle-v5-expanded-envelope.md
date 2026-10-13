# Goal-settle V5: training completion tilt envelope

## Why
At V4 update ~800?920 (2026-10-13), live monitor showed repeated samples parked ~2?4 cm from target with very low planar velocity, both wheel contacts and **~7 degrees body tilt**. V4 required <=0.08 rad (4.58 deg) for **both** completion reward and real retarget event. Capped settling-quality reward also had max tilt 0.08 rad, so it paid exactly zero. V4 nevertheless began occasional settled completions and reached navigation curriculum stage 1; stage 2 and long-goal arrival remain unsolved.

## V5 changes (training only)
- **Real retarget event**: explicit `target_max_tilt_rad=0.16` (9.17 deg) for its near-goal 0.5s dwell. Its underlying low-motion/support predicate now accepts an optional tilt threshold; default remains 0.08 rad for v1-v4 and other uses. The recovery-push stability predicate remains unchanged.
- **Angular matching:** V5 event uses XY angular speed <=0.25rad/s and yaw rate <=0.25rad/s separately, matching the existing completion bonus instead of the old event-only combined angular norm. V1-V4 remain unchanged.
- **Completion reward**: matching `max_tilt_rad=0.16`; retains one-shot bonus 30, 0.5s uninterrupted dwell, position <=0.10m, heading <=0.20rad, planar speed <=0.10m/s, both wheels.
- **Settling-quality shaping**: `max_tilt_rad=0.24` (13.75 deg), heading envelope 0.35rad, speed envelope 0.12m/s, angular envelope 0.40rad/s and yaw 0.35rad/s. Credit remains capped at 6 *total* per target attempt, not a farmable forever rate.
- No relaxation of the official/fixed evaluator limits, and no change to the navigation success counters that measure mere circle entry. Keep evaluating independent strict settled stops and long-goal arrival.
- All V4 navigation-first curriculum logic and anti-farming cost retained. Older tasks and checkpoints remain untouched.

## Follow-up
Run V5 as a separately signed reward task; a running V4 process cannot pick up these thresholds without restart. Preserve last V4 checkpoint/logs. Actor-only initialization from V4 is acceptable if explicitly marked as transfer from that same fresh lineage, with fresh critic and optimizer, to avoid resetting learned near-goal navigation. Verify real retarget transitions, not just completion reward. Monitor whether long-goal arrivals rise and whether widened training completions still pass the stricter evaluation gates. Roll back if the new envelope allows poor dynamic behavior.

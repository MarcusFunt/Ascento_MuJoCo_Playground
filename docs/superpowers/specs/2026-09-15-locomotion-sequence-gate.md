# Locomotion sequence gate design

## Goal

Make `Ascento-Locomotion-Flat` promotable only when a fixed-seed evaluator
proves settle, mild disturbance recovery, a nearby world-target step, and a
quiet stop. The frozen actor-only 79,999 transfer is the comparator; the
300-iteration locomotion checkpoint is a candidate, not a promotion.

## Design

The evaluator gains one reserved scenario command, `world_target_offset`. It
sets a world target relative to the robot's current yaw and position, so a
single suite remains valid across randomized reset positions and yaw. A
disturbance family applies its existing physical push first, then issues the
relative target step after a recovery window.

The runner records target arrival, arrival latency, final target error, and
post-arrival speed/heading. Existing recovery, survival, and stationary
quietness metrics remain the guardrails. `target_arrived` is summarized as a
binary metric, enabling a Wilson-LCB gate like survival and recovery.

## Acceptance

`locomotion_sequence_gate_v1` must require survival, recovery, target arrival,
final target error, post-target speed, post-target heading, and stationary
tilt/action-quality limits. Both the 79,999 transfer and `model_299` run on
the same immutable scenarios. Only a candidate that passes is eligible for
comparison; a failure is replayed and repaired by its implicated metric alone.

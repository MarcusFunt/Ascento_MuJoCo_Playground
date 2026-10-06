# Waypoint Heading Diversity Sweep

## Goal

Improve long-range, multi-waypoint driving when a checkpoint's requested final heading differs from the direction of travel, while retaining the current gate-hold benefits.

## Evidence and scope

The current gate-hold checkpoint improves waypoint gates over the long-range parent but still fails the three-metre turn family, sequence tilt, quiet-quality, and one hard Balance gate. Long-range training currently sets every regular target heading to the bearing toward its target. Gate-like episodes retain the reset heading and retarget forward after the push.

Change only the fraction of regular targets whose final heading is offset by exactly +90 or -90 degrees from their travel bearing. Compare fractions 0%, 12.5%, and 25%. Keep target distances and positions, the 25% gate-like hold mix, push/retarget timing, rewards, observations, action interface, PPO configuration, environment count, and seed fixed.

## Starting point and evaluation

Start each arm from the same actor-only transfer of the preserved gate-hold `model_299.pt` checkpoint. Use seed 73, 512 environments, 5,000 iterations, and fresh optimizer state for every arm. Evaluate each arm on `locomotion_sequence_gate_v1`, `locomotion_waypoint_gate_v2`, `balance_quiet_quality_regression_v1`, and `balance_gate_v5`, comparing identical suite/scenario hashes. Extend only the strongest heading-mixture arm if it improves the waypoint suite over the 0% control without worsening the sequence or Balance hard-gate counts.

Keep the full sweep under a nine-hour wall-clock cap. The existing gate-hold checkpoint remains an experiment and is never replaced.

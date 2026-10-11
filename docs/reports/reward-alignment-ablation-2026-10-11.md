# Generalist reward-alignment ablation

**Status:** complete. No reward treatment is promoted.

**Decision:** retain the selected actor-only locomotion transfer as the quality floor, keep the goal curriculum at stage 0, and do not continue training from any of the tested final checkpoints.

The complete run IDs, checkpoint paths and hashes, evaluation IDs, and paired comparisons are recorded in [reward_alignment_v1.json](../experiments/reward_alignment_v1.json).

## Protocol

Each arm started from the same selected locomotion actor (SHA-256 prefix cf68c195, full hash in the JSON record). The 41D actor was lifted to the 42D generalist observation with a zero-weight obstacle_mode input; all runs kept mode 0. Each arm had an actor-only transfer whose reward contract matched that arm. The critic, optimizer, iteration, and simulator state were fresh.

The reward comparison used seed 20261012, 512 environments, 300 updates, fixed goal-mix stage 0, unchanged safety/posture terms, and the same task/action/plant contracts. The arms were:

- Control: current velocity-progress reward and arrival weight 5.
- Bonus 8: arrival weight 8.
- Settled quality: smooth settled-quality weight 8.
- Potential progress: potential shaping scale 20, using the timestep-scaled transition signal.

The immutable hard gate contains 256 scenarios. The diagnostic suite contains 320 scenarios across precision-short, recovery/retarget-short, navigation-short, navigation-medium, and navigation-long cohorts. All diagnostic reports passed their report-consistency checks; diagnostic PASS is not a promotion gate.

Settled-quality used the trainer's default checkpoint cadence and therefore saved update 250 instead of updates 100 and 200. Updates did not change because of this; its update-299 checkpoint was evaluated against the common final-update outcomes, and update 250 was also screened.

## Hard-gate results

Final target error is the scenario p95. Recovery and arrival columns are Wilson lower confidence bounds. The formal final-error limit is 0.10 m.

| Policy checkpoint | Gate | Final error p95 | Recovery LCB | Arrival LCB | Main failure |
|---|---:|---:|---:|---:|---|
| Selected transfer | PASS | 0.56 cm | 0.985 | 0.985 | None |
| Control 100 | PASS | 1.04 cm | 0.985 | 0.985 | None |
| Control 200 | FAIL | 6.81 cm | 0.985 | 0.985 | Body rocking p95 0.0322 rad (limit 0.0300) |
| Control 299 | FAIL | 17.69 cm | 0.985 | 0.985 | Final error and stationary tilt |
| Bonus 8, 100 | PASS | 3.86 cm | 0.985 | 0.985 | None |
| Bonus 8, 200 | FAIL | 15.04 cm | 0.985 | 0.985 | Final error |
| Bonus 8, 299 | FAIL | 19.63 cm | 0.541 | 0.985 | Recovery, error, post-arrival speed/heading, tilt, and rocking |
| Settled quality, 250 | FAIL | 4.49 cm | 0.000 | 0.985 | Recovery, post-arrival speed/heading, tilt, action variation, and rocking |
| Settled quality, 299 | FAIL | 3.12 cm | 0.000 | 0.985 | Recovery, post-arrival speed/heading, tilt, action variation, and rocking |
| Potential progress, 100 | FAIL | 7.31 cm | 0.000 | 0.148 | Recovery, arrival, speed, and heading |
| Potential progress, 200 | FAIL | 6.05 cm | 0.000 | 0.000 | Recovery and arrival |
| Potential progress, 299 | FAIL | 8.32 cm | 0.000 | 0.000 | Recovery, arrival, tilt, and rocking |

The control is a strong retention warning: the hard gate passes at update 100, then target-error p95 grows from 1.04 cm to 17.69 cm by update 299, even though survival, recovery, and target-arrival lower bounds stay at 0.985. The arrival event by itself does not preserve precision.

## Diagnostic results by target band

Values below are final target-error p95 in meters. Every trained policy had settled-stop LCB 0 in all five diagnostic cohorts. The selected transfer had settled-stop LCB 0.943 in precision-short, recovery/retarget-short, and navigation-short; medium and long targets had no arrivals.

| Cohort | Transfer | Control 299 | Bonus 8, 299 | Quality 299 | Potential 299 |
|---|---:|---:|---:|---:|---:|
| Precision short | 0.0122 | 0.1803 | 0.2041 | 0.0126 | 0.1157 |
| Recovery/retarget short | 0.0064 | 0.1773 | 0.1965 | 0.0283 | 0.0869 |
| Navigation short | 0.0117 | 0.1799 | 0.2033 | 0.0127 | 0.1135 |
| Navigation medium | 0.7254 | 0.2346 | 0.7742 | 0.0547 | 0.2688 |
| Navigation long | 2.4324 | 2.4979 | 2.7307 | 2.4042 | 2.4224 |

All three short cohorts have target-arrival LCB 0.943 for control, bonus 8, and settled-quality; potential progress has 0 in those cohorts. Yet stopped-success LCB is 0 for every trained arm and cohort.

Settled-quality is the clearest tradeoff: it brings short and medium endpoints close to the target, but heading p95 reaches 1.50 rad in precision-short, 2.82 rad in recovery/retarget-short, and 2.53 rad in navigation-medium. Recovery LCB is 0, and stationary tilt/action-variation/rocking gates fail. It improves distance while teaching the wrong overall behavior.

The bonus-8 policy reaches short targets but leaves about 20 cm final error and never meets the settled-stop criterion. Its recovery/retarget recovery LCB is 0.599. Potential progress never records a target arrival in any cohort; short-target final-error p95 remains 11.4-11.6 cm.

## Paired comparison and training telemetry

All final diagnostic comparisons paired the same 320 scenario IDs against the selected transfer. Deltas are candidate minus transfer; positive final-error and heading deltas are worse.

| Arm | Final-error IQM delta (95% CI) | Heading-RMS IQM delta (95% CI) | Settled-stop IQM delta |
|---|---:|---:|---:|
| Control | +0.1317 m [0.1187, 0.1440] | +0.0089 rad [0.0054, 0.0129] | -0.70 |
| Bonus 8 | +0.1892 m [0.1885, 0.1897] | +0.2380 rad [0.2192, 0.2552] | -0.70 |
| Settled quality | -0.0131 m [-0.0209, -0.0074] | +1.7403 rad [1.7359, 1.7439] | -0.70 |
| Potential progress | +0.0557 m [0.0455, 0.0660] | +0.5025 rad [0.4699, 0.5323] | -0.70 |

The four 300-update runs completed with finite telemetry and repeated PPO updates. Final scalar returns were 315.4 (control), 331.0 (bonus 8), 402.1 (settled quality), and 178.0 (potential progress); these values are not comparable because each arm changes the reward scale. They do not track policy quality. Entropy fell substantially during training, while KL remained nonzero; the observed pattern is active policy drift with shrinking exploration, not a stalled process.

## Conclusion

The evidence supports a reward-alignment problem, but these three reward edits do not fix it:

- Raising the arrival bonus preserves arrivals temporarily, then worsens final precision and recovery.
- Adding settled quality improves endpoint distance but sacrifices heading, recovery, and stationary stability.
- Potential shaping at scale 20 suppresses target arrivals and recovery rather than supplying useful distance guidance.

No treatment passed the final hard gate. Increasing target distances from a failed update-299 policy would confound the reward comparison, so the curriculum continuation was not run. Keep the curriculum at stage 0 and the selected transfer as the rollback floor.

The potential-shaping test follows the policy-invariance result of [Ng, Harada, and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf), with the project's timestep scaling. The target-band decision follows the competence-proximal sampling idea in [Florensa et al. (2018)](https://proceedings.mlr.press/v80/florensa18a.html). A multi-critic approach such as [RobotKeyframing (CoRL 2025)](https://proceedings.mlr.press/v270/zargarbashi25a.html) remains a larger follow-up, not supported as the next change by this ablation.

# Quantitative evaluation

The evaluator is the project’s acceptance system. It is headless, vectorized,
versioned, deterministic by default, and independent of training rewards.

## Rules of evidence

1. A published suite is immutable. Make a new version if a reset range,
   scenario generator, event timing, metric definition, or threshold changes.
2. A policy passes only when every hard gate passes.
3. An evaluator inconsistency is `INVALID`, regardless of apparent policy
   performance.
4. A directory with only `suite.json` and resolved scenarios is `INCOMPLETE`;
   it does not supply a gate result.
5. Optional dense shaping is training instrumentation. Strict recovery and
   landing metrics remain the evaluation authority.

## Current suites

| Suite | Task | Intended use |
| --- | --- | --- |
| `balance_dev_v1` | Balance | Smaller development screen before a full gate |
| `balance_dev_v2` | Balance | Current development screen; also rejects yaw drift and persistent spin |
| `balance_gate_v1` | Balance | Original authoritative balance baseline |
| `balance_gate_v2` | Balance | Legacy pre-plant-contract balance gate; retained for historical artifacts |
| `balance_gate_v3` | Balance | Prior 65 Nm balance gate; retained for comparisons before world-target drift gating |
| `balance_gate_v4` | Balance | Prior world-target balance gate; adds a 120-second maximum target-error limit |
| `balance_gate_v5` | Balance | Current directional world-target gate; adds reset-heading and yaw-rate limits |
| `balance_recovery_edge_v1` | Balance recovery | 256-case fast screen concentrated on +0.08..+0.15 rad pitch with expanded roll/pitch-rate and translation ranges |
| `velocity_gate_v1` | Velocity | Deterministic twist/height command timelines |
| `locomotion_sequence_gate_v1` | Locomotion | Historical 15 cm body-relative target after a push; retained unchanged for comparison |
| `locomotion_waypoint_gate_v2` | Locomotion | Fixed world-map waypoints from 15 cm to 3 m, heading changes, a three-goal route, pushes, and sustained stop quality |
| `recovery_gate_v1` | Recovery | Wide-reset recovery, strict success, time, and continuous hold |
| `jump_gate_v1` | Jump | Takeoff, landing, recovered landing, distance, pre-impact speed, clearance, and hold |

The TOML files in `benchmarks/suites/` are authoritative for counts,
distributions, capabilities, and thresholds. `ascento evaluate suites --json`
returns a compact machine-readable inventory.

## Running an evaluation

Evaluate one explicit checkpoint or resolve the latest usable checkpoint from a
managed run ID:

```bash
ascento evaluate run --checkpoint logs/rsl_rl/.../model_2999.pt \
  --suite balance_gate_v5 --batch-size 512 --device auto --render-clips

ascento evaluate run --run-id <run-id> --suite recovery_gate_v1
```

`--checkpoint` and `--run-id` are mutually exclusive. A non-passing but valid
evaluation returns exit code 2 so automation cannot mistake it for success.

Screen several checkpoints against one suite before the full review:

```bash
ascento evaluate screen 'logs/rsl_rl/.../model_*.pt' \
  --suite balance_dev_v2 --top 3
```

Compare completed evaluations only when their stored scenario identities and
plant contracts are compatible. Artifacts without a plant contract are legacy
and are intentionally not comparable:

```bash
ascento evaluate compare <baseline-evaluation> <candidate-evaluation> \
  --output comparison.json
```

## Result states

| State | Meaning | Decision |
| --- | --- | --- |
| `PASS` | Evaluation was valid and all hard gates passed | Eligible for selection |
| `FAIL` | Evaluation was valid but one or more hard gates missed | Diagnose before retraining/promoting |
| `INCOMPLETE` | A required capability was missing, or an evaluation artifact did not finish | Do not infer policy quality; repair/re-run |
| `INVALID` | Results contradicted evaluator consistency checks | Fix evaluator/task lifecycle and re-run |

Some gates are documented as non-hard diagnostics, such as the current balance
saturation heuristic. They are visible in reports but do not by themselves
reject a policy.

## What is measured

The evaluator materializes each scenario deterministically from the suite ID,
root seed, family ID, and scenario index. It owns exact reset application,
command timelines, push disturbances, horizon accounting, and outcome
collection.

Examples of task metrics include:

- balance: survival, recovery, tilt, planar speed, world-position and
  reset-heading error, yaw rate, displacement, separately
  commanded/actuator-output/joint-applied effort, applied-effort saturation,
  recovery time, and leg symmetry;
- velocity: survival plus velocity and height tracking error;
- recovery: strict binary success, time from start, and stable-hold duration;
- jump: takeoff, landing, recovered landing, post-landing hold, distance error,
  landing speed, clearance, and shaping magnitude.

### World-waypoint gate

`locomotion_waypoint_gate_v2` is a simulation gate with 240 fixed-seed scenarios
in five 48-case families. A `world_target_pose = [x, y, yaw]` command sets an
absolute target in each clone's local flat-world map frame: XY is relative to
that clone's simulation origin and yaw is the world heading. The evaluator's
exact reset starts at the current pose as the hold target. The 15 cm family
retains the older gate's 4-second push, 9-second target command, and 20-second
horizon; longer families command 1 m, 2 m, and 3 m targets. The 3 m target asks
for a 90-degree final heading. The route family commands three distinct map
waypoints and applies a push between its second and third commands.

For this suite, a waypoint completes only after **0.75 seconds continuously**
within 0.10 m position error, 0.20 rad heading error, 0.08 m/s planar speed,
0.08 rad tilt, 0.25 rad/s roll/pitch and yaw rate, with both wheels supported.
Every commanded waypoint must complete before the next one is counted. The
final target must be inside that same envelope for **1.0 second immediately
before the horizon**. The gate also measures the full last 2 seconds after the
final command. A brief radius crossing therefore cannot establish route
completion or a stable stop. Failed/short episodes receive a finite failing
value for stop-window numeric metrics, so percentiles cannot omit them.

Each family gates survival (`Wilson lower >= 0.85`), full waypoint sequence and
final stopped state (`Wilson lower >= 0.80`), final position (`p95 <= 0.10 m`),
heading (`p95 <= 0.20 rad`), and speed (`p95 <= 0.08 m/s`). The last 2 seconds
also gate position, heading, speed, tilt (`p95 <= 0.05 rad`), body rocking
(`p95 <= 0.03 rad/s`), raw-action rate (`p95 <= 0.05`), raw-action second
difference (`p95 <= 0.18`), and high-frequency action-power ratio
(`p95 <= 0.10`, 10-Hz low-pass residual). The near-push family additionally
gates recovery (`Wilson lower >= 0.80`). These are frozen waypoint-stop limits;
the high-frequency ratio uses a continuous-filter **final-window** definition,
whereas the quiet-quality suite samples stationary periods. Evaluate the
quiet-quality and `balance_gate_v5` suites separately before selecting a model.

Run a checkpoint against the waypoint suite with a GPU batch that fits 12 GiB:

```bash
ascento evaluate run --checkpoint /absolute/path/model.pt \
  --suite locomotion_waypoint_gate_v2 --batch-size 48 --device cuda:0
```

The consistency audit verifies scenario/result identity and count, integer step
timing, end-of-horizon semantics, finite values, physical effort bounds, metric
fractions, displacement/path consistency, and successful episodes reaching the
requested horizon. A horizon timeout before the requested scenario horizon is a
lifecycle problem, not successful survival.

## Artifact contract

Each completed evaluation has an immutable directory under `evaluations/`:

```text
<evaluation-id>/
├── manifest.json                  provenance, compiled plant + checkpoint contract, source revision, device
├── suite.json                     immutable suite snapshot
├── resolved_scenarios.jsonl       exact materialized inputs
├── results.sqlite                 scenario outcomes and long-form metrics
├── summary.json                   family summaries
├── consistency.json               evaluator invariant results
├── gate.json                      hard/non-hard gate decisions
├── failures.json                  selected failure modes/scenarios
├── report.html                    portable human review
└── clips_manifest.json + clips/   optional rendered capture evidence
```

Use the CLI to inspect and archive rather than manually guessing paths:

```bash
ascento evaluate list --limit 50 --json
ascento evaluate report <evaluation-id> --json
ascento evaluate archive <evaluation-id>
```

The archive command creates a ZIP below the evaluation root and refuses paths
outside that root. It makes a copy; it does not remove the source evidence.

## Clips and exact replay

`--render-clips` makes a small capture set after the bulk headless pass. Clip
rendering is intentionally separate from the complete scenario pass so video
cost does not change the gate calculation. Inspect the clip manifest before
claiming that clips exist.

For a frame-by-frame diagnosis of one measured scenario, replay its stored ID:

```bash
ascento tools replay-evaluation -- <evaluation-id> \
  --scenario balance_gate_v5/disturbance/000000 --viewer native
```

The replay uses the stored resolved scenario. Restart it to return to that exact
initial state; an interactive viewer reset is intentionally not scenario-aware.

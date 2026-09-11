# Ascento and Wheel-Legged-Lab: topology comparison

This is a source-level comparison, not a claim that the two robots can share a
policy, gain, reward coefficient, or acceptance threshold. It was reviewed
against Wheel-Legged-Lab revision
[`e61bfe1`](https://github.com/zyicome/Wheel-Legged-Lab/tree/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7)
and this repository's task, controller, evaluator, CLI, MCP, and dashboard
sources. The projects solve adjacent problems with meaningfully different
plants and simulation stacks.

## Executive conclusion

Ascento has the stronger operational and evidence topology: one public action
contract, immutable plant/evaluation artifacts, exact scenario replay,
checkpoint provenance, managed runs, CLI/MCP parity, and a deliberately small
four-task graph. Wheel-Legged-Lab has the stronger *learning curriculum*
topology: an explicit skill ladder and a command formulation that makes heading
and yaw a first-class controlled quantity.

The appropriate transfer is therefore selective:

1. retain Ascento's MuJoCo plant, structured target controller, exact suites,
   and evidence-first model selection;
2. retain the new reset-relative XY + yaw target for Balance, rather than
   importing Wheel-Legged-Lab's Isaac-specific virtual-model controller;
3. make task topology part of the checkpoint ABI, so an observation/reward
   migration cannot silently load an old policy; and
4. surface the structured target diagnostics that the trainer already writes.

The implementation in this change does items 3 and 4. It intentionally does
not import VMC, terrain/depth/obstacle stages, Wheel-Legged-Lab reward weights,
or its telemetry-only stage-promotion gates.

## System topology

| Boundary | Ascento MuJoCo Playground | Wheel-Legged-Lab | Consequence |
| --- | --- | --- | --- |
| Simulator | MuJoCo through mjlab/Warp-backed execution | Isaac Lab / Isaac Sim | Controller and contact behaviour are not interchangeable. |
| Robot abstraction | Guard2 MJCF with a six-channel structured target ABI | A wheel-legged robot configured around a virtual-model controller | Joint target scales, wheel signs, torque limits, and reward coefficients cannot be copied. |
| Public policy action | Six normalized values: four leg position targets and two signed wheel velocity targets | Six values interpreted by VMC as virtual-leg angle/length and wheel-related commands | Both have six inputs, but identical tensor width is not compatibility. |
| Low-level actuation | Leg PD + wheel PI on each physics substep; output then passes motor response, torque-speed envelope, and limits | VMC maps virtual-leg quantities into joint/motor commands | Ascento must preserve its physical controller contract. |
| Task graph | Balance, Velocity, Recovery, Jump share plant/controller and specialize config | Ten named stages from flat locomotion through recovery, terrain, jumping, landing, moving/targeted obstacles | Wheel-Legged-Lab is a broad curriculum graph; Ascento is a deliberately narrow validated graph. |
| Evaluation authority | Immutable suites, fixed scenarios, gates, JSON/SQLite/HTML results, exact replay and capture | Training runs and stage progression are primary evidence | Ascento should not choose a checkpoint solely from training timeout/reward. |
| Operations | Managed artifact root, dashboard, CLI, MCP, maintenance/profile tooling | Training scripts/configuration centred on Isaac workflows | Ascento is suited to reproducible local research operations. |

### Ascento control/data path

```text
PPO policy (6 normalized values)
  -> structured_targets_v1
  -> leg position targets + signed wheel velocity targets
  -> substep leg PD / wheel PI
  -> motor response + torque-speed envelope + effort limits
  -> MuJoCo plant
  -> observations / rewards / metrics
  -> TensorBoard + training log + managed artifacts
  -> dashboard, CLI, MCP, deterministic evaluator and capture
```

The action contract records the order, scales, wheel signs, gains, integral
limit, request limit, and controller version. The plant contract records the
physics profile and MJCF hash. The added task contract records the policy-facing
observations, commands, rewards, terminations, reset/event terms, and metrics.

### Wheel-Legged-Lab control/data path

```text
PPO policy (virtual-leg / wheel command values)
  -> VMC action interpretation
  -> Isaac Lab robot and terrain simulation
  -> command tracking, posture, contact, and termination rewards
  -> training telemetry
  -> moving-mean stage gate / next-stage checkpoint load
```

Its [flat environment configuration](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/wheel_legged_gym/envs/wheel_legged/wheel_legged_flat_env_cfg.py)
and [reward implementation](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/wheel_legged_gym/envs/wheel_legged/mdp/rewards.py)
make this VMC- and robot-specific boundary explicit.

## Commands, directionality, and heading

### What Wheel-Legged-Lab does well

Its command term samples a command containing forward velocity, yaw rate,
height, and heading. Heading is converted to a wrapped heading error and then
to a bounded yaw-rate objective. This gives the policy both a directional goal
and a physically actionable short-horizon correction. The configuration also
includes heading tracking alongside velocity/yaw-rate tracking; see its
[command term](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/wheel_legged_gym/envs/wheel_legged/mdp/commonds.py).

### Ascento before and after the directional target change

Balance originally supplied a reset-relative planar position target only. A
point can constrain drift while leaving yaw under-specified: a robot can face
the point, rotate around it, or spin at it without changing the positional
score. The current Balance task now captures target XY and target yaw at reset,
observes body-frame XY error plus wrapped heading error, rewards heading, and
tracks a bounded yaw-rate correction derived from that heading error. Settled
balance also requires a small heading error and all base angular-rate axes.

Velocity remains a commanded twist/height task. Recovery and Jump deliberately
remove balance-only world target terms rather than inheriting an accidental
orientation objective. This is preferable to one global command object with
different unused fields because each policy's observation ABI stays explicit.

### Decision

Adapt the *idea* of heading as a first-class command signal, not the VMC
mapping. Ascento's reset-relative world target is appropriate for stationary
balance and can later become a gate/lap interface. Velocity should gain
heading-command variants only when there is an immutable velocity evaluation
suite that needs them; do not add a dormant heading channel merely for symmetry.

## Observations and privileged state

| Topic | Ascento | Wheel-Legged-Lab | Assessment |
| --- | --- | --- | --- |
| Actor state | Gravity, base velocity, height, joint state, contacts/forces, measured effort, previous action, reset-relative target error and heading error for Balance | Base state, gravity, command, joint state, virtual-leg kinematics, wheel velocity, previous action | Both use compact proprioception. Wheel-Legged-Lab's virtual-leg features match its VMC; Ascento should only add derived signals when a deterministic ablation proves value. |
| Critic state | Actor terms plus world root position/linear/angular velocity | Task-specific Isaac configuration, often richer state | Ascento's privileged critic is explicit and does not leak into deployment-facing actor observations. |
| Noise/randomization | Conservative simulation-only scope | More stage-dependent terrain/perception/randomization context | Avoid importing visual/depth inputs until flat-ground policies and gate evidence justify the additional ABI. |
| Target visibility | Target error is an actor input for Balance | Command/heading is an actor input | Comparable at the goal level, but not in coordinate system or controller meaning. |

The newly added task topology contract is important here. Tensor shape alone
does not establish semantic compatibility: two observations can have the same
dimension while ordering, scaling, or a reward objective changes. Every new
checkpoint now embeds the task fingerprint; resume, evaluation, replay, and
capture reject missing or incompatible topology before rollout.

## Reward topology

Ascento Balance combines alive/upright/height, position and heading retention,
angular/planar motion penalties, leg symmetry/hold, measured physical effort,
controller-request barrier, and target-command rate regularization. The
position/yaw target is global only in the reset-relative sense: it is invariant
to where the parallel environment happens to be placed in the tiled world.

Wheel-Legged-Lab combines posture, base-height, velocity/yaw tracking, heading,
leg/wheel smoothness, torque, collisions, nominal pose, and termination terms.
The high-level reward families overlap, but numerical weights and Gaussian
scales depend on its robot, units, VMC mapping, contact model, and stage. They
are hypotheses, not defaults for Ascento.

The key current distinction is command versus physics:

- target position/velocity magnitude is a policy-command diagnostic;
- `controller_requested_effort` is a low-level PD/PI request;
- actuator output and joint-applied effort are measured downstream physical
  quantities; and
- position drift/heading error are task outcomes.

These must not be folded into a single metric called “effort” or “action.”

## Curriculum and training strategy

Wheel-Legged-Lab documents a ten-stage sequence (flat, recovery, terrain
reactivity, jumping, landing, clearance, moving targets, target landing, and
obstacle stages) with moving-average training gates and checkpoint transfer.
See [its staged-training guide](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/STAGED_TRAINING.md).

Ascento currently has two kinds of progression:

- separate task configs for Balance, Velocity, Recovery, and Jump; and
- a Balance/Velocity horizon schedule of 20, 60, 120, then 300 seconds. The
  runner promotes after sustained timeout windows, protects the final horizon
  from automatic demotion, lowers the final-stage learning rate, and saves a
  candidate. Deterministic gate evaluation, rather than the timeout rate,
  decides whether a candidate is usable.

This is intentionally less ambitious. The next safe curriculum improvement is
not terrain: finish the balance foundation under a directional target, compare
short pilots deterministically, then add controlled disturbance amplitude or
frequency only after stable 300-second target hold has been demonstrated. A
cross-task checkpoint transfer should be treated as a new experiment with an
explicit observation/action/task-contract compatibility decision, not assumed
because a neural-network layer happens to load.

## Evaluation, provenance, and repository operation

| Capability | Ascento | Wheel-Legged-Lab | Recommendation |
| --- | --- | --- | --- |
| Checkpoint identity | SHA-256 plus plant/action/task contracts | Stage checkpoint handoff | Keep Ascento's strict identity; it catches semantic drift before GPU time is spent. |
| Scenario evaluation | Immutable suite snapshot, resolved scenarios, consistency checks, gates, SQLite/HTML/JSON | Training-stage success signals | Never replace suites with training moving averages. |
| Replay/capture | Exact reset reconstruction, capture metadata, optional video | Isaac visualization/debug tooling | Keep exact replay as the source of truth for a visual claim. |
| Model selection | Candidate retained by training, then deterministic gate adjudication | Stage progression uses training telemetry | Use Wheel-Legged-Lab-style stage gates only as cheap *promotion candidates*, never as pass/fail evidence. |
| Operations | Native WSL profiles, Docker/dashboard, artifact roots, MCP/CLI, maintenance | Script/config-led workflows | Keep the native artifact root and detached-job workflow; do not move long runs back to a OneDrive-mounted artifact tree. |

## Dashboard observability audit

The dashboard is intentionally a bounded, live control plane, not an
everything viewer. “Available” below means the data exists in a current run,
artifact, evaluator, capture, or MCP/CLI response. It does not mean the web UI
currently renders a chart for it.

| Information | Available from | Dashboard status after this change | How to inspect when not shown |
| --- | --- | --- | --- |
| Reward, episode length, PPO/value loss, entropy, KL, clip fraction, learning rate, throughput | TensorBoard/log telemetry | Shown as canonical charts/cards | Dashboard telemetry endpoint, CLI/MCP run detail |
| Leg position-target RMS, wheel velocity-target RMS, controller-request saturation | TensorBoard and standard trainer log | **Now shown** as structured-target charts | Raw normalized telemetry retains the source tags |
| Controller request RMS/mean and measured actuator-output saturation | Trainer telemetry when emitted | Shown only when those tags exist | Raw telemetry and evaluation metrics; blank is unknown, not zero |
| Target XY error, heading error, yaw rate, tilt, root speed, per-joint request/output/applied effort | Environment metric manager and deterministic evaluator | Not a live dashboard chart by default | Evaluation `summary.json`/`results.sqlite`, capture state arrays, or MCP evaluation report |
| Per-reward-term returns and term contribution over time | Environment/reward manager when recorded | Not normalized into dashboard charts | TensorBoard tags or targeted instrumentation; do not infer from total reward |
| Raw policy action distribution, individual leg/wheel targets, wheel PI integral, PD/PI unsaturated request, motor-response state, torque-speed clipping | Controller/action runtime and capture-capable state | Not exposed as dashboard time series | Controller probes/capture artifacts; add a dedicated diagnostic capture rather than a high-rate dashboard stream |
| Contact events/forces, body pose/velocity trajectory, command history, reset state, termination reason per episode | Evaluator/capture artifacts | Not in the live dashboard | `results.sqlite`, resolved scenario JSONL, capture NPZ, exact replay/video |
| Gate results, confidence intervals, consistency checks, worst scenarios, rendered clips | Evaluation artifact set | Not on dashboard mutation/API pages | CLI/MCP `list_evaluation_reports` / `get_evaluation_report` |
| Full checkpoint provenance (plant/action/task contracts), package versions, config snapshot, raw launch argv | Checkpoint and experiment/evaluation manifests | Contract status is now shown on selected runs; full structures are not rendered | Manifest JSON, MCP run detail, evaluation manifest |
| Curriculum windows, promotion/demotion/protection, candidate checkpoint | Horizon runner status lines and run metadata | Current stage/window summary shown; full transition history is not charted | Training log and raw telemetry |
| GPU memory/utilization, process namespace/PID status, Tailscale/update state | Host/system probes | System page / run health subset | System endpoint, CLI/MCP health |

Two operating rules follow from this audit:

1. A dashboard blank never proves zero; it normally means the compatible source
   did not emit that tag or the view deliberately avoids a high-rate stream.
2. A dashboard trend never proves a model passes. Use deterministic evaluation
   artifacts and exact replay/capture for selection and diagnosis.

## Findings fixed in this review

1. **Task-ABI hole:** previously, a checkpoint recorded plant and action
   contracts but not its observation/reward/reset/termination topology. A
   changed task could therefore reach a generic model-load failure or, for
   same-shaped changes, silently resume under a new objective. New checkpoints
   embed `ascento_task_topology_v1`; legacy or incompatible checkpoints now
   fail clearly before resume, evaluation, replay, or capture.
2. **Incomplete artifact provenance:** experiment manifests, final checkpoint
   records, capture metadata, evaluation manifests, run comparison deltas, and
   dashboard run annotations now include the task contract alongside plant and
   action contracts.
3. **Structured-control blind spot:** the trainer already emitted leg-target
   RMS, wheel-target RMS, and controller-request saturation, but the dashboard
   parsed only legacy/general loss names and described commands as direct
   torque. The log parser, canonical metric mapping, charts, and wording now
   preserve the actual `structured_targets_v1` semantics.

## Deliberately deferred work

- No VMC port: it would replace a tested Ascento controller ABI with
  robot/simulator-specific dynamics.
- No obstacle/depth/terrain task import: there is no corresponding validated
  Ascento terrain/perception contract yet.
- No automatic stage selection based on dashboard reward or timeout alone:
  deterministic gates remain authoritative.
- No high-rate controller internals streamed to the web dashboard: that would
  turn the monitoring plane into an unbounded telemetry system. Capture and
  targeted probes are the correct evidence path.

## Follow-on experiments

1. Train a short directional-target Balance pilot from scratch and evaluate
   multiple saved checkpoints with the current Balance gate before adjusting
   gains or weights.
2. Compare heading error, yaw-rate RMS, target drift, timeout rate, wheel target
   RMS, and controller-request saturation together. Do not accept a longer
   timeout if it comes from unbounded drift or spin.
3. If the pilot reliably holds target XY + yaw, add a fixed, low-amplitude
   disturbance schedule and repeat the same deterministic gate comparison.
4. Only then design a gate command interface for locomotion. It should define
   position, traversal direction, tolerance, and completion semantics before
   reward weights are introduced.

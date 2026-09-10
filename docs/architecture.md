# Architecture

Ascento MuJoCo Playground is a simulation-only motion-authoring project for an
Ascento Guard-2-like wheel-legged robot. It uses mjlab's MuJoCo Warp runtime and
RSL-RL PPO. It is not a physical-robot controller, and its deliberately high
simulation authority must not be read as a hardware specification.

## Runtime boundaries

```text
robot.xml
  -> robot_cfg.py + actuator implementation
  -> mjlab environment/task managers
  -> RSL-RL PPO trainer
  -> managed run artifacts
  -> Dashboard / CLI / MCP / deterministic evaluator / captures
```

| Layer | Owns | Primary source |
| --- | --- | --- |
| Robot asset | Kinematic and physical MJCF model | `src/ascento_mjlab/assets/ascento_guard2/robot.xml` |
| Plant contract | Joint names, supported pose, actuator configuration, simulator timing | `robot_cfg.py`, `actuator.py`, `actuator_impl.py`, `physics.py` |
| mjlab task layer | Scene, actions, observations, commands, resets, events, rewards, terminations, metrics, and recording | `src/ascento_mjlab/tasks/`, `src/ascento_mjlab/mdp/` |
| Optimizer | PPO rollout, optimization, logging, and checkpoints | mjlab/RSL-RL configuration in `tasks/*/rl_cfg.py` |
| Project services | Run lifecycle, provenance, dashboard telemetry, CLI, MCP, evaluation, clips | `dashboard/`, `cli.py`, `mcp_server.py`, `evaluation/`, `tools/` |

The project does not retain a second active simulation backend. Historical
JAX/MJX/Brax work is diagnostic history only; it is not an acceptance target.

## Plant and timing contract

The robot has six normalized `structured_targets_v1` actions in this fixed order:

- left hip position, left knee position, left wheel velocity;
- right hip position, right knee position, right wheel velocity.

Leg targets are offsets of up to pi/2 from the nominal -pi pose and are clamped
to the MJCF limits. Wheel targets span +/-8 rad/s; both wheel-joint signs are
inverted so equal positive policy commands move forward in the base-frame
convention. A positive left target with a negative right target turns clockwise
(negative base yaw) when viewed from above. `ascento tools controller-probe`
checks those physical conventions, neutral 20-second hold, and indexed wheel-PI
reset behavior before a balance training run.
A physics-rate PD controller
for legs and PI controller with conditional anti-windup for wheels produce a
requested torque. The
canonical `animation_high_authority` physics profile uses a 0.002 s MuJoCo
timestep, decimation of 5 (0.01 s control period), a 0.75 m supported root
height, and 65 Nm peak simulated effort. The motor stage then applies a linear
torque-speed envelope, controller-speed protection, and finite response time.
Leg and wheel continuous-torque figures in `actuator.py` are documentation-only
figures, not thermal limits.

No communication delay, sensor noise, thermal model, or hardware safety model
is enabled. Do not use a high simulated command-saturation value as a
hardware-safety conclusion; it is a diagnostic for this simulator.

## MDP responsibilities

`src/ascento_mjlab/mdp/` holds task-specific terms rather than a parallel
environment framework:

| Module | Responsibility |
| --- | --- |
| `commands.py` | Height and compound motion commands |
| `events.py` | Supported-pose initialization and training events |
| `observations.py` | Policy/critic observations, including command state |
| `rewards.py` | Balance, velocity, effort, jump, and shaping terms |
| `recovery.py` | Recovery envelope, progress, dwell shaping, and strict success state |
| `jump.py` | Jump-state synchronization and event state |
| `terminations.py` | Fall, velocity, and non-finite termination semantics |
| `metrics.py` | Diagnostic terms recorded with the task |

Rewards guide learning; they are not the pass/fail contract. The deterministic
evaluation suites independently calculate task and physical metrics, then apply
their versioned gates.

## Task topology

All current tasks are flat-ground tasks and share the same base plant:

1. `Ascento-Balance-Flat` establishes supported, disturbance-tolerant balance.
2. `Ascento-Velocity-Flat` adds twist and height commands while retaining the
   stabilizing plant.
3. `Ascento-Recovery-Flat` specializes wide-reset and push recovery.
4. `Ascento-Jump-Flat` adds a single motion command, jump state, takeoff,
   flight, landing, and post-landing recovery.

Terrain, obstacles, slopes, and high-landing variants remain sequenced behind
quantitatively sound flat-ground jump behavior. Do not introduce terrain as a
shortcut around a weak flat-ground policy.

## Artifact and control-plane topology

The Dashboard, CLI, and MCP server share the filesystem-backed
`dashboard.run_service.RunService`; they do not proxy one another over HTTP.
This keeps managed-run metadata, stop semantics, provenance, and artifact-root
selection consistent across interfaces.

```text
managed launcher -> logs/rsl_rl/<task>/<run>/
                       |-- run_metadata.json / status / console / TensorBoard
                       `-- checkpoints

deterministic evaluator -> evaluations/<evaluation-id>/
                            |-- immutable suite and resolved scenarios
                            |-- SQLite results, summaries, gate decision
                            `-- HTML report and optional clips
```

`ASCENTO_ARTIFACT_ROOT` selects the managed-run root; it defaults to
`logs/rsl_rl`. `ASCENTO_EVALUATION_ROOT` selects the evaluation root; it
defaults to `evaluations`. Both are resolved relative to the checkout when
given as relative paths.

## Change boundaries

- Update `physics.py`, the actuator implementation, robot configuration, and
  tests together when changing a plant-wide physical value.
- Update a task configuration and its reward/metric implementation together.
- Do not silently edit a published benchmark. Create a new suite version when a
  scenario distribution, metric definition, threshold, or event timing changes.
- Treat evaluator behavior, task metrics, and horizon completion as one
  lifecycle contract; a good training curve cannot compensate for an ambiguous
  evaluator result.

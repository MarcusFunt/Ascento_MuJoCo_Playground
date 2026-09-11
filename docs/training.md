# Training policies

Training is staged PPO optimization, not evidence of acceptance. A checkpoint
becomes a candidate because it trains; it becomes a selected policy only after
the relevant immutable evaluation suite passes.

## Prepare the environment

Use the maintained Linux/WSL2 path from [Operations](operations.md). For manual
development, use exactly one compute extra:

```bash
uv sync --frozen --extra cu128 --extra dashboard
# or, without a CUDA GPU:
uv sync --frozen --extra cpu --extra dashboard
```

The `cpu` and `cu128` extras are mutually exclusive. Keep the extra on `uv run`
commands when the active environment is not already configured for it.

## Stage order and task purpose

| Stage | Task | Learns | Training-only conditions | Acceptance suite |
| --- | --- | --- | --- | --- |
| 1 | `Ascento-Balance-Flat` | Supported balance near a per-environment world target, controlled effort, recovery from planar pushes | Curriculum through 20, 60, 120, and 300 s; a cardinal push between 4–6 s | `balance_gate_v4` |
| 2 | `Ascento-Velocity-Flat` | Linear velocity, yaw-rate, and height tracking | Random twist/height resampling every 3–6 s | `velocity_gate_v1` |
| 3 | `Ascento-Recovery-Flat` | Wide-reset stabilization and recovery after a physical push | Broad initial roll/pitch/velocity envelope; interval push only during training | `recovery_gate_v1` |
| 4 | `Ascento-Jump-Flat` | Commanded crouch, takeoff, flight, landing, distance, and post-landing stabilization | Flat-ground compound motion command | `jump_gate_v1` |

The task configurations are the source of exact reward weights and reset
ranges. Training changes must be evaluated against their gates, not accepted
from scalar reward or episode length alone.

Before starting or materially changing balance training, run the deterministic
controller characterization on the intended compute backend:

```bash
ascento tools controller-probe -- --device cuda:0 --json
```

It records a neutral structured target for 20 seconds from the exact supported
pose, checks equal-wheel forward motion and opposing-wheel clockwise turn, and
confirms an indexed reset clears only the selected wheel PI state. The neutral
result is an open-loop baseline—not a policy gate for this dynamically balanced
robot. Treat a failed direction or PI-reset check as a plant/controller issue,
not a PPO tuning result.

## Starting a managed run

Use the unified CLI so that the dashboard and MCP can observe the run:

```bash
uv run --frozen --extra cu128 --extra dashboard ascento run start \
  --task Ascento-Balance-Flat \
  --display-name "balance baseline" \
  --purpose baseline \
  --tag balance --tag seed-123 \
  --envs 512 --iterations 10000 --seed 123
```

Use `--` to forward native trainer options that do not have first-class CLI
flags:

```bash
uv run --frozen --extra cu128 --extra dashboard ascento run start \
  --task Ascento-Velocity-Flat --display-name "velocity candidate" \
  -- --agent.save-interval 250 --agent.logger tensorboard
```

`--parent-run-id` and `--parent-checkpoint` record lineage. Use them whenever a
run continues or branches from a previous policy. Metadata can be corrected
without moving artifacts:

```bash
ascento run annotate <run-id> --purpose ablation --tag no-dense-shaping
```

To continue a managed checkpoint, start a child run. Pass its exact checkpoint
as `--parent-checkpoint` and forward the three resume options below. The
launcher creates its read-only resume link before the trainer starts, so do not
try to copy or link checkpoint directories after `run start` returns:

```bash
ascento run start --task Ascento-Balance-Flat \
  --display-name "balance continuation" --parent-run-id <parent-run-id> \
  --parent-checkpoint /absolute/path/to/model_7999.pt \
  --envs 512 --iterations 16000 --seed 321 -- \
  --agent.resume True --agent.load-run _resume_parent \
  --agent.load-checkpoint model_7999.pt
```

`--iterations` is the number of additional PPO iterations after loading; it is
not a new total. Keep the parent checkpoint in place for the child run's
duration because its optimizer state is loaded from that file at startup.

## Reward shaping and strict metrics

Recovery and jump include dense shaping to make sparse events learnable:

- recovery uses progress plus a stable-dwell signal;
- jump uses phase-aware task rewards plus post-landing stabilization.

Set `ASCENTO_DISABLE_DENSE_SHAPING=1` for an ablation. This changes training
rewards only; it does not weaken the strict binary recovery and landing metrics
in the evaluator. Report shaping magnitude alongside task reward so a dense
term cannot silently mask a binary regression.

Balance tuning can be explored with these explicit environment variables:

- `ASCENTO_BALANCE_DRIFT_PENALTY_SCALE`;
- `ASCENTO_BALANCE_STABILIZATION_WEIGHT`;
- `ASCENTO_BALANCE_PUSH_INTERVAL_MIN_S`; and
- `ASCENTO_BALANCE_PUSH_INTERVAL_MAX_S`.

Balance initializes a separate absolute XY target for every cloned environment
at its supported reset pose. The actor receives the target error in its yaw
frame, while the reward measures world-frame distance. This permits the wheels
to move whenever doing so reduces true target error or prevents a fall; only a
small instantaneous speed regularizer remains alongside action-rate and
physical-effort penalties. Future navigation may advance the same world target
through a gate sequence without changing the observation or reward interface.

Record any non-default values in the run notes/tags and experiment manifest
metadata. Do not promote a policy based on an override whose effect was not
measured against the default acceptance suite.

## Monitor without distorting the workstation

For a live run, use bounded, low-frequency reads rather than repeatedly loading
the entire dashboard or log:

```bash
ascento run monitor <run-id> --interval 30
ascento run progress <run-id> --json
ascento run telemetry <run-id> --limit 50 --json
ascento run logs <run-id> --tail 100
```

`progress` is the cheap latest snapshot. `telemetry` intentionally bounds its
history. The dashboard retains the whole run span by sampling large histories,
so chart density is not a measure of training refresh cadence. `run compare`
also contrasts task/config ID, seed, environment count, simulator timestep,
device, dense-shaping state, reward terms, overrides, and evaluation metadata
against its first run, in addition to normalized metrics.

Watch trends, not individual PPO loss spikes. Episode length can plateau at a
configured horizon; it is not a proof of recovery or gate success. Empty
advanced-diagnostic cards mean no compatible telemetry was emitted, not a zero
value.

## Long-run readiness

Before a substantial run, execute the production preflight with actual balance,
velocity, and recovery checkpoints:

```bash
export ASCENTO_BALANCE_CHECKPOINT=logs/rsl_rl/ascento_balance/<run>/model_<n>.pt
export ASCENTO_VELOCITY_CHECKPOINT=logs/rsl_rl/ascento_velocity/<run>/model_<n>.pt
export ASCENTO_RECOVERY_CHECKPOINT=logs/rsl_rl/ascento_recovery/<run>/model_<n>.pt
export ASCENTO_COMPUTE_EXTRA=cu128
bash scripts/preflight_long_run.sh
```

The script runs finite-plant/model checks, the test suite, a two-iteration
512-environment PPO smoke, and deterministic evaluator preflight. It exits
nonzero when required checkpoint variables are absent. Do not start a long run
when it fails.

## Selecting and continuing a policy

For balance, screen candidate checkpoints—including
`model_best_long_horizon.pt` when it exists—on the development suite first,
then use `balance_gate_v4` for the final decision. The curriculum writes that
checkpoint after its first qualifying 300-second window and replaces it only
with a better qualifying window. At 300 seconds, PPO uses a fixed `1e-5`
learning rate to protect an already-stable controller from adaptive-rate
regression. The best-horizon checkpoint is a candidate, not a pass.

```bash
ascento evaluate screen 'logs/rsl_rl/ascento_balance/<run>/model_*.pt' \
  --suite balance_dev_v1 --top 3
ascento evaluate run --checkpoint <selected-checkpoint> --suite balance_gate_v4
```

Use a new managed run for a material change in rewards, curriculum, seed,
environment count, simulator, or source revision. Preserve lineage instead of
overwriting prior evidence.

# `ascento` CLI reference

`ascento` is the supported command-line control plane for managed training,
evaluation, capture, dashboard control, maintenance, and MCP serving. It shares
the dashboard's filesystem-backed run service, so a run created here has the
same stable ID, metadata, status, and stop behavior as a dashboard-created run.

For agents, the project-local [Codex CLI skill](../.codex/skills/ascento-cli/SKILL.md)
adds operation-selection and safety guidance.

## Invocation

Run from the checkout. Use the dashboard extra for run management and the CUDA
extra for simulator-heavy work:

```bash
uv run --frozen --extra dashboard ascento <command>
uv run --frozen --extra cu128 --extra dashboard ascento <command>
```

Use `--extra cpu` instead of `--extra cu128` for CPU-only work. `--json` emits
structured JSON suitable for scripts and agents. Paths that are not absolute are
resolved against the repository root unless the command states otherwise.

## Command map

```text
ascento
├── run       start, inspect, monitor, compare, annotate, stop managed training
├── evaluate  run, screen, compare, inspect, archive immutable evaluations
├── capture   capture state channels and optional policy MP4 clips
├── tools     forward to specialist clip, quality, reward, and replay tools
├── dashboard start or query the local FastAPI dashboard
├── maintain  forward arguments to scripts/maintain.sh
└── mcp serve run the stdio MCP server
```

## `ascento run`

Run subcommands share:

| Option | Meaning |
| --- | --- |
| `--artifact-root PATH` | Managed-run root; defaults to `ASCENTO_ARTIFACT_ROOT` or `logs/rsl_rl` |
| `--json` | Machine-readable output |

### `run start`

```text
ascento run start [options] [-- <native trainer arguments>]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--task ID` | `Ascento-Balance-Flat` | Registered mjlab task |
| `--name NAME` | — | Compatibility alias for display name |
| `--display-name NAME` | task ID | Human-readable run name |
| `--purpose TEXT` | `exploratory` | Experiment intent stored in metadata |
| `--tag TAG` | repeatable | Replaceable list of classification tags |
| `--notes TEXT` | empty | Free-text run notes |
| `--parent-run-id ID` | — | Parent managed run for lineage |
| `--parent-checkpoint PATH` | — | Parent checkpoint for lineage |
| `--envs N` | — | Adds `--env.scene.num-envs N` to trainer args |
| `--iterations N` | — | Adds `--agent.max-iterations N` to trainer args |
| `--seed N` | — | Adds `--agent.seed N` to trainer args |
| `--episode-horizon-s S` | — | Records configured horizon metadata |
| `--foreground` | false | Poll until completion instead of returning after creation |
| `--interval S` | `15.0` | Foreground polling interval |

Everything after `--` is forwarded unchanged to the mjlab trainer. Prefer the
first-class options when available because they also create clear metadata.

Examples:

```bash
ascento run start --task Ascento-Recovery-Flat \
  --display-name "recovery dense-shaping ablation" \
  --purpose ablation --tag recovery --tag no-dense-shaping \
  --envs 512 --iterations 4000 --seed 31

ascento run start --task Ascento-Balance-Flat --display-name "long candidate" \
  -- --agent.save-interval 250 --agent.logger tensorboard
```

### Read and monitor runs

| Command | Required input | Extra options | Result |
| --- | --- | --- | --- |
| `run list` | none | `--active` | All discovered runs, or only `starting`/`running`/`stopping` runs |
| `run status RUN_ID` | run ID | — | Detailed health, provenance, metadata, and telemetry |
| `run progress RUN_ID` | run ID | — | Cheap latest progress snapshot for polling |
| `run monitor RUN_ID` | run ID | `--interval S` (30), `--once` | Prints only changed progress snapshots until terminal state |
| `run logs RUN_ID` | run ID | `--tail N` (200) | Bounded training-log tail |
| `run telemetry RUN_ID` | run ID | `--limit N` (50) | Bounded normalized telemetry history |
| `run compare RUN_ID RUN_ID [RUN_ID …]` | 2–8 unique run IDs | — | Normalized latest metrics plus experiment-manifest/configuration deltas |

Use `progress` for a low-cost heartbeat and `telemetry` only when recent metric
history is needed. `monitor --once` is useful in scripts and agents.

### Change run metadata or stop a run

```text
ascento run annotate RUN_ID [metadata fields]
ascento run stop RUN_ID [--reason TEXT]
```

`annotate` accepts any nonempty combination of `--display-name`, `--notes`,
`--purpose`, repeatable `--tag`, `--parent-run-id`, and `--parent-checkpoint`.
Supplying `--tag` replaces the complete tag list rather than appending to the
existing metadata. It does not rename or move an artifact directory.

`stop` requests a graceful stop through the managed process group. Its default
reason is `user_requested`; use a short factual reason when automation stops a
run. It may escalate only after the launcher has had time to clean up.

## `ascento evaluate`

Evaluation subcommands share:

| Option | Default | Meaning |
| --- | --- | --- |
| `--artifact-root PATH` | managed artifact root | Needed when resolving `--run-id` |
| `--output-root PATH` | `ASCENTO_EVALUATION_ROOT` or `evaluations` | Evaluation artifact root |
| `--json` | false | Structured output |

### `evaluate run`

```text
ascento evaluate run (--checkpoint PATH | --run-id RUN_ID) --suite ID_OR_TOML [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint PATH` | mutually exclusive | Exact model/checkpoint file |
| `--run-id RUN_ID` | mutually exclusive | Resolve latest usable managed-run checkpoint |
| `--suite ID_OR_TOML` | required | Suite ID in `benchmarks/suites` or explicit TOML path |
| `--batch-size N` | `512` | Number of scenarios per vectorized evaluator batch |
| `--device DEVICE` | `auto` | `cuda:0`, `cpu`, or another torch device |
| `--render-clips` | false | Capture a small optional clip set after evaluation |
| `--clip-takes N` | `3` | Capture takes when rendering clips |
| `--clip-steps N` | `600` | Steps per rendered capture |

The command returns zero only for `PASS`, returns 2 for a valid non-pass, and
prints the completed artifact details. It must not be used to infer a policy
result from a partial directory.

### `evaluate screen`

```text
ascento evaluate screen CHECKPOINT_OR_GLOB [CHECKPOINT_OR_GLOB …] --suite ID [options]
```

It expands checkpoint paths/globs, evaluates each usable checkpoint, calculates
the screening score, and prints the top rows. Options: `--suite` (required),
`--batch-size` (256), `--device` (`auto`), and `--top` (3). Use it to narrow a
candidate set, then run the authoritative suite on selected checkpoints.

### Compare, inspect, and archive evaluation evidence

| Command | Arguments | Extra options | Result |
| --- | --- | --- | --- |
| `evaluate compare` | `BASELINE CANDIDATE` | `--output PATH` | Paired deltas for compatible stored scenarios; optionally writes JSON below output root |
| `evaluate list` | none | `--limit N` (50) | Newest complete and incomplete report summaries |
| `evaluate report` | `EVALUATION` | — | Manifest, gates, consistency, failures, suite, and clips payload |
| `evaluate archive` | `EVALUATION` | `--output FILE.zip` | ZIP copy of one evaluation directory; output must remain below output root |
| `evaluate suites` | none | — | Immutable suite inventory |

`EVALUATION` can be an ID relative to the output root or an absolute directory
below it. Archive creation is recoverable: it overwrites/creates the ZIP but
does not delete source evidence.

### `evaluate preflight`

```text
ascento evaluate preflight \
  --balance-checkpoint PATH --velocity-checkpoint PATH --recovery-checkpoint PATH [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| three checkpoint options | required | Actual balance, velocity, and recovery candidates |
| `--device DEVICE` | `auto` | Evaluator device |
| `--max-scenarios N` | `64` | Bounded scenarios per preflight suite |
| `--batch-size N` | `64` | Preflight vector batch size |
| `--output PATH` | `preflight/evaluator.json` | JSON written relative to checkout unless absolute |

Preflight checks deterministic evaluator behavior across the three core task
families. It is a readiness check, not a replacement for a full suite.

## `ascento capture`

```text
ascento capture (--checkpoint PATH | --run-id RUN_ID) [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint PATH` / `--run-id RUN_ID` | one required | Source policy |
| `--task ID` | inferred for a run; balance otherwise | Registered task |
| `--takes N` | `3` | Number of recorded takes |
| `--steps N` | `1000` | Control steps per take |
| `--output-dir PATH` | timestamped `captures/…` | NPZ capture directory |
| `--video-dir PATH` | unset | Write optional rendered MP4 clips there |
| `--device DEVICE` | `auto` | Torch/simulator device |
| `--artifact-root PATH` | default root | Needed for run lookup |
| `--json` | false | Structured capture manifest |

The output contains state channels and metadata (task, seed, checkpoint hash,
physics profile, action/effort, contacts, and event state where applicable).
Capture is visual evidence, not a benchmark gate.

## `ascento tools`

Specialist commands receive all arguments after an optional `--` unchanged.
Use the separator so their flags cannot be parsed as outer CLI options.

| Command | Underlying module | Arguments |
| --- | --- | --- |
| `tools clip-motion -- INPUT.npz` | `tools.clip_motion` | `--output PATH`, `--fps FPS`, `--event all|takeoff|landing`, `--pre-roll S` (0.5), `--post-roll S` (0.5) |
| `tools rank-motion -- INPUT…` | `tools.motion_quality` | one or more NPZ files/directories; `--top N`, `--output PATH` |
| `tools reward-probe` | `tools.reward_probe` | no arguments; prints deterministic reward-geometry checks |
| `tools replay-evaluation -- EVAL_DIR` | `evaluation.replay` | `--scenario ID` required, `--checkpoint PATH`, `--viewer native|viser`, `--device DEVICE` |

Examples:

```bash
ascento tools clip-motion -- captures/run/take_000.npz --event landing --fps 24
ascento tools rank-motion -- captures/run --top 5 --output captures/run/ranking.json
ascento tools replay-evaluation -- evaluations/<id> \
  --scenario recovery_gate_v1/recovery_random/000000 --viewer native
```

## Dashboard, maintenance, and MCP

### `ascento dashboard`

| Command | Options | Behavior |
| --- | --- | --- |
| `dashboard status` | `--host` (`127.0.0.1`), `--port` (8000), `--timeout` (5), `--json` | Reads `/api/health`; returns nonzero if unreachable |
| `dashboard start` | `--host`, `--port`, `--foreground`, `--reload`, `--json` | Starts `uvicorn dashboard.app:app`; detached by default |

### `ascento maintain`

```text
ascento maintain [-- <maintain.sh arguments>]
```

This invokes `bash scripts/maintain.sh` from the repository root. See
`bash scripts/maintain.sh --help` and [Operations](operations.md) for exact
options. Maintenance can modify packages, Docker images, and the checkout; it
will refuse dirty tracked state unless its own `--force` option is supplied.

### `ascento mcp serve`

Starts the project’s stdio MCP server. It writes protocol traffic to stdout, so
do not wrap it in tools that add regular log output to stdout. Use the Windows
registration script described in [Operations](operations.md) for Codex Desktop,
or run this directly for a generic MCP client:

```bash
uv run --frozen --extra dashboard --extra mcp ascento mcp serve
```

See [MCP reference](mcp-reference.md) for the available tools and their state
effects.

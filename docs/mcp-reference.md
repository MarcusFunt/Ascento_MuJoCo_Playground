# MCP reference

The Ascento MCP server is a stdio integration over the same filesystem-backed
run service used by the dashboard and CLI. It is intended for low-frequency,
bounded monitoring and deliberate operational calls from agents.

Start it with `ascento mcp serve`, or register the WSL launcher using
`scripts/register_codex_mcp.ps1` as described in [Operations](operations.md).
The server's stdout is reserved for MCP JSON-RPC.

## Tool selection

Use the lowest-impact read that answers the question:

| Need | Preferred tool |
| --- | --- |
| Is a run live and moving? | `get_run_progress` |
| What happened recently? | `get_run_telemetry` or `get_run_logs` with a small bound |
| Full provenance/health/checkpoint state | `get_run_details` or `get_run_checkpoint` |
| Dashboard/repository health | `get_dashboard_health` |
| Gate result or artifacts | `list_evaluation_reports`, then `get_evaluation_report` |
| Run an expensive policy decision | `evaluate_policy` only when synchronous completion is wanted |

The machine-readable [mcp-tools.json](mcp-tools.json) contains complete input
names, defaults, mutability, and cost classification.

## Read tools

| Tool | Inputs | Output |
| --- | --- | --- |
| `list_runs` | `active_only=false` | Run summaries; optional active-state filter |
| `get_run_progress` | `run_id` | Cheap latest progress/health snapshot |
| `get_run_details` | `run_id` | Detail, provenance, telemetry, health, metadata |
| `get_run_logs` | `run_id`, `tail=200` | Bounded log tail (1–5000 lines) |
| `get_run_telemetry` | `run_id`, `limit=50` | Bounded normalized records (1–2000) |
| `get_dashboard_health` | none | Dashboard health and repository-version status |
| `compare_runs` | `run_ids` | Latest normalized metrics across 2–8 runs |
| `get_run_checkpoint` | `run_id` | Latest usable managed-run checkpoint |
| `list_evaluation_suites` | none | Immutable suite IDs, task, scenario/family/gate counts |
| `list_evaluation_reports` | `output_root`, `limit=50` | Complete and incomplete report summaries |
| `get_evaluation_report` | `evaluation`, `output_root` | Gate, consistency, failures, suite, clip metadata |
| `compare_evaluation_reports` | `baseline`, `candidate`, `output_root` | Paired stored-scenario deltas |

## Mutation and long-running tools

| Tool | Effect | Important inputs |
| --- | --- | --- |
| `start_run` | Starts managed training | task, metadata/lineage, envs, iterations, seed, trainer args |
| `stop_run` | Requests graceful stop | run ID, reason |
| `update_run_metadata` | Updates sidecar metadata only | any nonempty metadata/lineage field |
| `evaluate_policy` | Runs full deterministic evaluation and writes evidence | exactly one of checkpoint/run ID, suite, batch/device, optional clips |
| `archive_evaluation_report` | Creates ZIP copy of an evaluation | evaluation, optional root/output |
| `capture_policy` | Writes NPZ channels and optional MP4 clips | exactly one of checkpoint/run ID, task/take/step/output controls |
| `run_evaluator_preflight` | Runs bounded deterministic core-task preflight | balance, velocity, recovery checkpoints, device/bounds |

`evaluate_policy` is intentionally synchronous. It can hold the caller until
the complete result exists and should not be used as a frequent heartbeat.
`capture_policy` and rendering can be expensive. Do not start, stop, or update
runs without the user's authorization.

## Artifact roots and incomplete reports

The server uses `ASCENTO_ARTIFACT_ROOT` for managed runs and
`ASCENTO_EVALUATION_ROOT` for evaluations. `output_root` overrides only the
evaluation root for that invocation.

The report listing deliberately includes a directory with `suite.json` but no
`manifest.json` as `INCOMPLETE`; this is evidence of an interrupted or still
running evaluator, not a hidden passing result. Inspect the directory or rerun
the evaluation before judging the checkpoint.

## Registration notes

The PowerShell registration helper validates a WSL distribution and registers
the native WSL shell launcher with the Codex CLI. It explicitly assigns the
server's managed-run root to
`/root/Ascento_MuJoCo_Playground/logs/rsl_rl`; MCP reads and run operations then
refer to the same artifacts as the native WSL CLI. The native profile must be
maintained before registration. It does not register tools in an already-running
conversation; open a new task or restart Codex Desktop afterward.

> [!WARNING]
> Run `scripts/register_codex_mcp.ps1` from PowerShell, but do not run `uv`
> there against the WSL checkout. That selects Windows Python and can fail while
> manipulating the WSL project's Linux `.venv`. Use an Ubuntu/WSL shell for all
> `uv sync` and `uv run` commands.

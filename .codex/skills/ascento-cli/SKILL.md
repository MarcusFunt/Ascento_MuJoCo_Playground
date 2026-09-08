---
name: ascento-cli
description: Operate and inspect the Ascento MuJoCo Playground through its unified CLI. Use for managed training runs, quantitative evaluation, policy capture, dashboard checks, maintenance, or the project MCP server; do not use for general MuJoCo questions unrelated to this checkout.
---

# Ascento CLI

Use the project CLI from the repository root when working on Ascento operations.
It shares the dashboard's filesystem-backed run service, so it is the preferred
way to create, inspect, compare, annotate, stop, evaluate, capture, and archive
artifacts without recreating path or process-management logic.

For complete current flags, defaults, output behavior, and specialist-tool
arguments, read [the CLI reference](../../../docs/cli-reference.md). For a
machine-readable command inventory, use
[docs/cli-reference.json](../../../docs/cli-reference.json). Treat
`src/ascento_mjlab/cli.py` as the executable source of truth if a discrepancy is
found.

## Environment selection

- Use `uv run --frozen --extra dashboard ascento …` for run service and
  dashboard operations.
- Add `--extra cu128` for simulator-heavy evaluation, capture, or GPU training;
  use `--extra cpu` instead on a CPU-only host. Never select both compute extras.
- Prefer `--json` for programmatic/agent reasoning.
- Resolve run artifacts through `--artifact-root` or `ASCENTO_ARTIFACT_ROOT`;
  resolve evaluator evidence through `--output-root` or
  `ASCENTO_EVALUATION_ROOT`.

## Choose the smallest operation

| Need | Command |
| --- | --- |
| List active runs | `ascento run list --active --json` |
| Cheap live status | `ascento run progress <run-id> --json` |
| Changed-only terminal monitor | `ascento run monitor <run-id> --interval 30` |
| Recent metrics or logs | `ascento run telemetry <run-id> --limit 50 --json` or `ascento run logs <run-id> --tail 100` |
| Full run provenance/health | `ascento run status <run-id> --json` |
| Compare experiments | `ascento run compare <id-a> <id-b> --json` |
| Inspect/record lineage | `ascento run annotate <run-id> …` |
| Resolve/evaluate a candidate | `ascento evaluate run --run-id <id> --suite <suite>` |
| Inspect gate results | `ascento evaluate list --json`, then `ascento evaluate report <evaluation> --json` |
| Render state/video evidence | `ascento capture --run-id <id> …` or evaluator `--render-clips` |
| Exact failure replay | `ascento tools replay-evaluation -- <evaluation-dir> --scenario <id>` |
| Dashboard reachability | `ascento dashboard status --json` |

Use `run progress` rather than polling the GUI or full logs for routine
monitoring. Poll no more often than every 30–60 seconds unless the user needs a
specific immediate transition. Bounded telemetry/log requests are the next
choice; a full evaluator or capture is not a heartbeat mechanism.

## Managed run behavior

`run start` accepts first-class metadata (`--display-name`, `--purpose`,
repeatable `--tag`, `--notes`, parent run/checkpoint) and common trainer fields
(`--envs`, `--iterations`, `--seed`). Put unmodeled native trainer arguments
after `--`. Use first-class fields where available because they populate durable
metadata.

Use `run annotate` to change metadata without moving an artifact directory.
Passing one or more `--tag` arguments replaces the tag list. Use `run stop`
only with user authorization; it requests graceful trainer shutdown through the
managed process group.

## Evaluation behavior

Use exactly one of `--checkpoint` and `--run-id` for `evaluate run` and
`capture`. A non-passing valid evaluation returns exit code 2; do not treat that
as a command malfunction. Evaluate a checkpoint with an immutable suite before
calling it acceptable. Training reward, horizon-reaching episode length, and
clips alone are not pass criteria.

An evaluator directory with `suite.json` but no `manifest.json` is
`INCOMPLETE`, usually because it was interrupted or is still running. Do not
infer a gate result from it. Inspect with `evaluate report` or rerun after
repairing the underlying issue. Archive copies are constrained below the
evaluation root and do not delete source evidence.

Use `evaluate screen` to reduce a candidate set and `evaluate compare` only for
compatible completed evaluation artifacts. Use `evaluate preflight` before a
substantial run when real balance, velocity, and recovery checkpoints are
available.

## State-changing operations

Starting/stopping a run, changing run metadata, running an evaluation, creating
captures/clips/archives, and invoking maintenance all change local state or can
consume substantial compute. Require user authorization for those operations.
Do not start maintenance merely to inspect state; use dashboard/run commands
first. `ascento maintain` forwards to `scripts/maintain.sh`, which can rebuild
Docker and update the checkout and intentionally rejects dirty tracked state
unless its own `--force` is supplied.

## MCP handoff

`ascento mcp serve` starts the same project MCP server over stdio. Keep stdout
clean because it carries JSON-RPC. The Windows registration helper is
`scripts/register_codex_mcp.ps1`; after registration, a new Codex task or app
restart is required to acquire MCP tools. Read
[docs/mcp-reference.md](../../../docs/mcp-reference.md) before choosing an MCP
tool over the CLI.

## Documentation maintenance

When changing public CLI behavior, update both `docs/cli-reference.md` and
`docs/cli-reference.json`, then revise this skill if operation selection,
safety, or important semantics changed. Keep the CLI parser as the source of
truth and validate the documentation contract before committing.

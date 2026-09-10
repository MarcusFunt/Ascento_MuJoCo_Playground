# Troubleshooting

Start with bounded, source-backed checks. Do not diagnose a run from one chart
or a single reward value.

## Run is not visible in the dashboard

1. Check the dashboard's artifact root and the launcher's artifact root:

   ```bash
   ascento dashboard status --json
   ascento run list --active --json
   ```

2. Confirm `ASCENTO_ARTIFACT_ROOT` points at the same directory for both
   processes. The default is `<checkout>/logs/rsl_rl`.
3. If the dashboard is containerized, ensure the host `logs/` directory is
   mounted and the dashboard image has been rebuilt after source changes.
4. Inspect the managed run status and log tail before starting a replacement:

   ```bash
   ascento run status <run-id> --json
   ascento run logs <run-id> --tail 200
   ```

## Dashboard is slow or charts look incomplete

The dashboard samples large histories across their full span to bound browser
payloads. A chart that contains fewer points than the raw log should still show
the beginning and end of the run. Compare with the low-impact CLI reads:

```bash
ascento run progress <run-id> --json
ascento run telemetry <run-id> --limit 100 --json
```

For live training, monitor every 30–60 seconds instead of continually polling
the full page. On WSL-mounted workspaces, repeated filesystem scans are more
expensive than a cheap progress snapshot. If a deployed dashboard still behaves
like old source, run the maintained update path to rebuild the image:

```bash
bash scripts/maintain.sh
```

## Advanced diagnostics show `—` or empty charts

`—` means the current records lack that metric; it does not mean zero effort,
zero saturation, or zero recovery success. First inspect recent normalized
telemetry and the launcher log. Then check that the training code actually
emits the diagnostic and that the active Docker image/virtual environment uses
the same commit as the checkout. Do not repair the UI by plotting invented
defaults.

## Episode length or reward dips sharply

Treat an abrupt synchronized episode-length/reward drop as a signal to inspect
the underlying reset, curriculum, termination, horizon, and event data. A
timeout ceiling can produce a flat high line even if strict recovery is poor.
Use logs, normalized telemetry, and an exact evaluator scenario before changing
rewards. The evaluator’s end-of-horizon consistency checks are the authority for
survival/recovery claims.

## Evaluation is `INCOMPLETE` or `INVALID`

- `INCOMPLETE` with no `manifest.json` usually means the evaluation was
  interrupted or remains unfinished. Do not use it to judge the policy.
- `INCOMPLETE` with a manifest names missing required capabilities. Repair the
  task/evaluator capability before re-running.
- `INVALID` means a scenario/result/timing/physical consistency invariant
  failed. Preserve the artifact, inspect `consistency.json` and `failures.json`,
  then fix the lifecycle issue before training more.

Use `ascento evaluate report <id> --json` rather than manually inferring a
result from partial files.

## GPU or virtual-environment failures

Use one compute extra only. A partial or mixed Torch environment can produce
import or CUDA mismatches. Reconcile it from the locked project definition:

```bash
uv sync --frozen --all-groups --extra cu128 --extra dashboard --extra mcp
```

For CPU development, replace `cu128` with `cpu`. Run
`python -m ascento_mjlab.tools.smoke` after synchronization to confirm Torch,
CUDA/device, MuJoCo/Warp, and finite simulation steps.

## One-shot WSL command returned, but no run started

Do not infer success from a PID printed by `wsl.exe`. In this workstation's
Windows/WSL configuration, a command backgrounded from a one-shot invocation
can be terminated when that invocation returns, including a child wrapped in
`nohup`. This leaves no native run artifacts or training process.

Use the interactive, foreground procedure in
[Operations](operations.md#start-a-managed-cuda-training-run-on-this-workstation).
If a launch looks suspicious, check the native install before starting a
replacement:

```bash
cd /root/Ascento_MuJoCo_Playground
.venv/bin/ascento run list --active --json
```

An empty list means no managed run was created. If a run ID exists, inspect it
before deciding whether it needs a graceful stop or normal completion.

## Maintenance refuses to update

The maintainer correctly refuses a dirty tracked checkout or local-only commits
unless `--force` is given. Commit or intentionally preserve changes first. Do
not use `--force` merely to avoid inspecting unrecognized changes; it can
discard work. Active managed runs also block dashboard-initiated maintenance.

## MCP server is missing from Codex

From PowerShell, run `scripts/register_codex_mcp.ps1` (add `-Replace` only for
an intentional replacement). It needs the Codex CLI, `wsl.exe`, the named WSL
distribution, and the repository's Linux virtual environment. After a successful
registration, start a new Codex task or restart Codex Desktop. An existing task
cannot dynamically acquire the new tool list.

## Clips are missing or do not match the hypothesis

Bulk evaluation does not render every scenario. Request `--render-clips`, then
inspect `clips_manifest.json`. For an exact failure, use the stored scenario ID
with `ascento tools replay-evaluation`. Review the event/contact/action channels
frame-by-frame before changing controller, actuator, reward, or renderer code.

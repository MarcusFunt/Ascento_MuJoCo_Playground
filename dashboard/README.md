# Ascento Training Dashboard

A local web dashboard for monitoring mjlab/RSL-RL training over Tailscale. It reads
training artifacts and launcher metadata; it does not own the PPO algorithm itself.

## What it shows

- current iteration, wall-clock freshness, throughput, elapsed time and ETA;
- reward and episode-length trends;
- PPO/surrogate loss, value loss, entropy, KL, clip fraction and invalid-update telemetry when reported by the trainer;
- NaN/Inf detection across numeric telemetry;
- stale-run detection when a run is still marked running but telemetry stops;
- GPU utilization, memory, temperature, process PID/aliveness and per-process GPU memory when `nvidia-smi` is available;
- recent traceback/error excerpts and live console output;
- Git commit/branch, task/stage, seed, command line, timestep/device, checkpoint path, configuration files, start/end time and exit code;
- a copyable run-information block and downloadable `run-summary.json`;
- active dashboard configuration plus API health.

The dashboard and `dashboard.launch` use the same artifact-root configuration.
The default is:

```text
<repo>/logs/rsl_rl
```

Set `ASCENTO_ARTIFACT_ROOT` once to override it for both processes.

## Install

From the repository root:

```bash
uv sync --frozen --extra cu128 --extra dashboard
cd dashboard/frontend
npm install
npm run build
cd ../..
```

Use `--extra cpu` instead of `--extra cu128` on a CPU-only machine.

For frontend development, run `npm run dev` in `dashboard/frontend`; Vite proxies
`/api` requests to `127.0.0.1:8000`.

## Start the dashboard

Use the project interpreter through uv:

```bash
uv run --frozen --extra dashboard python -m uvicorn dashboard.app:app \
  --host 127.0.0.1 --port 8000
```

Or use:

```bash
./scripts/run_dashboard.sh
```

The production server intentionally binds only to `127.0.0.1:8000`. To make it
available to your tailnet without exposing it on the LAN:

```bash
tailscale serve --bg 8000
```

Open the HTTPS Tailscale Serve URL from another device on the same tailnet.
Do not use Tailscale Funnel for this private monitor.

To use a different artifact root:

```bash
ASCENTO_ARTIFACT_ROOT=/path/to/logs \
  uv run --frozen --extra dashboard python -m uvicorn dashboard.app:app \
  --host 127.0.0.1 --port 8000
```

The API exposes:

- `/api/health` — health indicator and startup/configuration diagnostics;
- `/api/config` — active dashboard configuration;
- `/api/runs` — discovered runs;
- `/api/runs/<id>` — detailed run status, health and reproducibility metadata;
- `/api/runs/<id>/summary.json` — downloadable run summary;
- `/api/runs/<id>/telemetry` — normalized training telemetry;
- `/api/runs/<id>/logs` and `/logs/stream` — captured console output.

`ASCENTO_STALE_AFTER_SECONDS` controls the stale-run threshold and defaults to
90 seconds.

## Start a monitored training run

Launch training through the same uv-managed environment:

```bash
uv run --frozen --extra cu128 python -m dashboard.launch \
  --task Ascento-Balance-Flat \
  -- --env.scene.num-envs 512 --agent.max-iterations 10000
```

`dashboard.launch` captures durable run metadata before starting the trainer,
including the current Git commit/branch, the exact command, task/stage, seed,
device and simulation timestep when supplied on the command line. It updates the
same status record with PID, finish time, exit code and latest checkpoint.

To override the shared root for both the launcher and server:

```bash
ASCENTO_ARTIFACT_ROOT=/path/to/logs \
  uv run --frozen --extra cu128 python -m dashboard.launch \
  --task Ascento-Balance-Flat \
  -- --agent.max-iterations 10000
```

For motion exports, keep using the normal post-training tooling, for example:

```bash
uv run --frozen --extra cu128 capture-motion \
  Ascento-Jump-Flat --checkpoint /path/to/model_10000.pt --takes 20
```

## Startup failures

The backend validates its artifact root during startup. If the configured path
is a file, cannot be created, or cannot be read/searched, startup fails with a
message naming the bad path and the relevant environment variable. A missing
frontend build is non-fatal and appears as a startup warning in `/api/health`.

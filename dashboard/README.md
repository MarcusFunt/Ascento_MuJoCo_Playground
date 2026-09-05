# Ascento Training Dashboard

A local web control plane for mjlab/RSL-RL training over Tailscale. The dashboard
monitors existing artifacts and can also start/stop managed training runs. PPO
still belongs to mjlab/RSL-RL; the dashboard owns run lifecycle, provenance,
console capture, health reporting, and comparison.

## Run management

The **Runs** page is now the default dashboard view. It supports:

- starting a training run from the browser;
- human-readable names independent of machine artifact directory names;
- notes, tags, purpose, parent-run lineage and parent-checkpoint provenance;
- editing metadata for old runs without moving or rewriting their artifacts;
- graceful stop requests for dashboard-managed processes;
- side-by-side comparison using normalized reward, episode-length, PPO-loss,
  entropy, KL and clip-fraction telemetry;
- explicit repository-version status for every run.

Managed runs are launched through `python -m dashboard.launch`, so browser and
CLI runs share the same metadata/status schema and console capture. The API
starts each managed launcher in a dedicated process session. A stop request marks
the run `stopping`, sends SIGINT to that process group, waits for trainer cleanup,
and only escalates to terminate/kill if the process does not exit.

Run metadata is stored next to the run as `run_metadata.json`. The artifact
directory remains machine-oriented and immutable; display names and notes can be
changed later without breaking paths.

## What the monitor shows

- current iteration, wall-clock freshness, environment-step throughput, elapsed
  time and ETA;
- explicit iteration counts plus collected environment-transition totals when
  run config is available;
- reward and episode-length trends;
- PPO/surrogate loss, value loss, entropy, KL, clip fraction and invalid-update
  telemetry when reported by the trainer;
- NaN/Inf detection across numeric telemetry;
- stale-run detection when a run is still marked running but telemetry stops;
- GPU utilization, memory and temperature, plus process-specific telemetry when
  the dashboard shares the trainer's PID namespace;
- recent traceback/error excerpts and live console output;
- Git commit/branch, task/stage, seed, command line, timestep/device, checkpoint
  path, configuration files, start/end time and exit code;
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
npm ci
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

Open the HTTPS Tailscale Serve URL from another device on the same tailnet. Do
not use Tailscale Funnel for this private control surface.

The maintained Docker dashboard mounts `logs/` read-write because managed runs
create their status, logs, TensorBoard files and checkpoints under the shared
artifact root. `checkpoints/`, `captures/`, and `evaluations/` remain read-only in
the dashboard service.

## API

Read/monitor endpoints:

- `GET /api/health`
- `GET /api/config`
- `GET /api/runs`
- `GET /api/runs/<id>`
- `GET /api/runs/<id>/summary.json`
- `GET /api/runs/<id>/telemetry`
- `GET /api/runs/<id>/logs`
- `GET /api/runs/<id>/logs/stream`

Run-control endpoints:

- `POST /api/runs` — start a managed run;
- `PATCH /api/runs/<id>` — edit name/notes/tags/purpose/lineage;
- `POST /api/runs/<id>/stop` — request a graceful stop;
- `GET /api/runs/compare?run_ids=<id1>,<id2>` — compare 2–8 runs.

The Runs page passes training CLI tokens as an explicit string list. In the UI,
enter one option/value per line so values containing punctuation are not
re-tokenized by a shell.

## CLI launch

The CLI remains useful for automation and records the same provenance:

```bash
uv run --frozen --extra cu128 python -m dashboard.launch \
  --display-name "Recovery baseline after PR83" \
  --purpose baseline \
  --tag recovery --tag pr83 \
  --task Ascento-Recovery-Flat \
  -- --env.scene.num-envs 512 --agent.max-iterations 10000
```

Optional lineage can be recorded with `--parent-run-id` and
`--parent-checkpoint`.

Native RSL-RL TensorBoard event steps are PPO iterations, not environment
transitions. Normalized telemetry therefore uses `iteration`,
`completed_iterations`, and `total_iterations` for progress. When
`num_steps_per_env` and `scene.num_envs` are available, it also reports
`environment_steps` and `total_environment_steps`. `Perf/total_fps` is exposed
as environment-step throughput rather than iteration throughput.

`ASCENTO_STALE_AFTER_SECONDS` controls the stale-run threshold and defaults to
90 seconds.

## Startup failures

The backend validates its artifact root during startup. If the configured path
is a file, cannot be read/searched, or later proves unwritable when starting a
run, the error names the artifact root. A missing frontend build is non-fatal
and appears as a startup warning in `/api/health`.

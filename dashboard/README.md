# Ascento Training Dashboard

A local web control plane for mjlab/RSL-RL training. The dashboard monitors
existing artifacts, starts/stops managed training runs, compares results, reports
repository state, and can request a guarded update to the newest `origin/main`.
PPO still belongs to mjlab/RSL-RL; the dashboard owns run lifecycle, provenance,
console capture, health reporting, comparison, and the user-facing control flow.

## Run management

The **Runs** page supports:

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

## Host supervisor and repository updates

The **System** page shows:

- checked-out branch and exact local commit;
- current `origin/main` commit, ahead/behind counts and incoming commit subjects;
- dirty working-tree state;
- active training runs that block maintenance;
- last/current update state;
- Tailscale sidecar state, MagicDNS name and Tailnet IPs.

The Dashboard container does **not** receive `/var/run/docker.sock`, a repository
mount, `sudo`, or an arbitrary command endpoint. A small host-side supervisor
runs as the workstation user and accepts only two operations over a Unix-domain
socket:

1. `status` — fetch/compare Git state and report update/Tailnet status;
2. `update` — invoke the repository's existing `scripts/maintain.sh` against
   `main` after safety checks pass.

Install that boundary once after updating the checkout:

```bash
bash scripts/install_supervisor.sh
```

The installer creates a boot-persistent systemd service. The process runs as the
workstation user, with Docker-group access only so `maintain.sh` can rebuild the
project stack. The Dashboard receives only `.maintenance/supervisor/supervisor.sock`
through a bind mount.

The supervisor refuses a GUI-triggered update when:

- the checkout is not on `main`;
- tracked files are modified;
- the checkout has local-only commits;
- `origin/main` cannot be refreshed;
- another update is running;
- any run is still `starting`, `running`, or `stopping`.

When accepted, the supervisor starts `maintain.sh` in an independent process
session and writes progress to `.maintenance/update-state.json` and
`.maintenance/update.log`. The Dashboard and its Tailnet sidecar may disappear
briefly while Docker rebuilds, but the supervisor remains alive outside Docker
and finishes the update. If Tailnet access was enabled, `maintain.sh` includes
the Tailscale Compose overlay again and the persistent node reconnects.

System-control API:

- `GET /api/system` — cached host/Tailnet/update state;
- `GET /api/system?refresh=true` — force a fresh `origin/main` check;
- `POST /api/system/update` — request the guarded maintenance update.

## Remote access through Tailscale

Only the Dashboard-facing service is intentionally exposed to the Tailnet. The
`ascento-mjlab` service is not attached to the Tailscale sidecar and no LAN or
public-internet port is opened.

Tailscale's official container runs as a sidecar and the Dashboard shares its
network namespace. Local workstation access remains available on
`127.0.0.1:8000`; Tailnet peers can connect to port `8000` on the node's
Tailscale IP or MagicDNS name.

Enroll the sidecar once:

```bash
bash scripts/setup_tailscale.sh
```

The script accepts a Tailscale auth key or OAuth client secret through the
hidden prompt, or through the `TS_AUTHKEY` environment variable. It uses that
credential only for initial enrollment. After Tailscale state has been persisted
in the `tailscale-dashboard-state` Docker volume, the sidecar is immediately
recreated without the credential so the secret is no longer present in Docker's
inspectable container environment.

Optional hostname:

```bash
ASCENTO_TAILSCALE_HOSTNAME=ascento-workstation \
  bash scripts/setup_tailscale.sh
```

After enrollment the script prints the remote URL, normally similar to:

```text
http://ascento-dashboard.<tailnet-name>.ts.net:8000
```

The HTTP payload travels inside Tailscale's encrypted tunnel; this configuration
does not enable Tailscale Funnel and does not expose the Dashboard publicly.
Your normal Tailnet grants/ACLs still determine which users/devices may reach the
node.

The Tailscale sidecar uses kernel networking (`/dev/net/tun`) and persistent
state. `scripts/maintain.sh` remembers enrollment via the ignored local marker
`.maintenance/tailscale-enabled` and automatically includes
`docker/compose.tailscale.yaml` on later rebuilds. No auth key is stored in the
repository.

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

The maintained path is:

```bash
bash scripts/maintain.sh
bash scripts/install_supervisor.sh
bash scripts/setup_tailscale.sh   # optional, but required for remote Tailnet GUI access
```

For a manual development environment:

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

## API

Read/monitor endpoints:

- `GET /api/health`
- `GET /api/config`
- `GET /api/system`
- `GET /api/runs`
- `GET /api/runs/<id>`
- `GET /api/runs/<id>/summary.json`
- `GET /api/runs/<id>/telemetry`
- `GET /api/runs/<id>/logs`
- `GET /api/runs/<id>/logs/stream`

Control endpoints:

- `POST /api/system/update` — guarded update to newest `origin/main`;
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
and appears as a startup warning in `/api/health`. A missing host supervisor does
not prevent monitoring/run management; the System page simply disables host
updates until `scripts/install_supervisor.sh` has been run.

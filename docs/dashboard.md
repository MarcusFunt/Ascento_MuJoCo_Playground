# Dashboard and HTTP API

The dashboard is a local control plane for artifact-backed RSL-RL/mjlab runs.
It monitors existing artifacts, starts and stops managed runs, compares
normalized telemetry, reports provenance and health, and can ask the guarded
host supervisor to update the checkout.

## Pages and data semantics

### Runs

The Runs page lists managed and discovered runs, exposes display metadata,
lineage, notes, tags, current health, console tails, and side-by-side metric
comparison. A run directory remains machine-oriented; `run_metadata.json`
stores mutable presentation metadata beside it.

The monitor surfaces:

- iteration, completion percentage, environment steps, throughput, elapsed time,
  ETA, and freshness;
- reward, episode length, PPO/surrogate loss, value loss, entropy, KL, and clip
  fraction when the trainer emits them;
- structured-target diagnostics (leg position-target RMS, wheel velocity-target
  RMS, controller-request saturation), controller/actuator effort diagnostics,
  recovery success/time/hold, and shaping terms when compatible telemetry is
  emitted;
- NaN/Inf, stale-run, recent-error, GPU, checkpoint, task, seed, command-line,
  and Git provenance data.

Interpretation rules:

- A reward trend is an optimization signal, not a benchmark result.
- Episode length reaching a ceiling often means configured timeout, not a
  policy that recovered or passed its gate.
- PPO loss is an objective term, not a score to minimize visually.
- A blank advanced-diagnostic card means the source record does not contain
  that metric. It is not equivalent to zero.
- Target velocity is a command; controller-request torque is a PD/PI request;
  measured actuator output is downstream physical behaviour. The dashboard
  does not treat these as interchangeable direct-torque actions.
- The chart endpoint samples oversized histories across the full run span; a
  short-looking segment can be a payload limit or missing telemetry, not
  necessarily a recent regression.

### System

The System page reports the current checkout and `origin/main`, ahead/behind
state, dirty state, active runs that block maintenance, update state, and
Tailscale status. Update actions are delegated to the host supervisor; the web
container cannot run an arbitrary command or access Docker's socket.

## HTTP endpoints

FastAPI publishes OpenAPI at `/openapi.json` while the service is running. The
human-oriented endpoint list is:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Dashboard health and startup warnings |
| `GET` | `/api/config` | Public runtime configuration |
| `GET` | `/api/system` | Cached host/update/Tailnet status |
| `POST` | `/api/system/update` | Request guarded maintenance update |
| `GET` | `/api/runs` | List run summaries |
| `POST` | `/api/runs` | Start a managed run |
| `GET` | `/api/runs/compare?run_ids=…` | Compare 2–8 runs |
| `GET` | `/api/runs/{id}` | Full run detail |
| `GET` | `/api/runs/{id}/progress` | Cheap latest progress snapshot |
| `PATCH` | `/api/runs/{id}` | Update display metadata and lineage |
| `POST` | `/api/runs/{id}/stop` | Request graceful stop |
| `GET` | `/api/runs/{id}/summary.json` | Downloadable run summary |
| `GET` | `/api/runs/{id}/telemetry` | Normalized telemetry (`limit`, optional `max_points`) |
| `GET` | `/api/runs/{id}/logs` | Bounded log tail |
| `GET` | `/api/runs/{id}/logs/stream` | Live server-sent log stream |

The REST API does not expose evaluator or capture mutation. Use the CLI or MCP
for those operations so all artifact handling follows the same project contract.

## Deliberately non-dashboard evidence

The dashboard remains bounded live monitoring. It does not chart every
per-episode evaluator result, capture state channel, contact trajectory,
controller PI state, per-joint request/output signal, reward-term contribution,
or full checkpoint/manifest object. Those are available through evaluation
artifacts, capture NPZ files, exact replay, and the CLI/MCP report tools. See
[the comparison's observability audit](wheeled-legged-lab-comparison.md#dashboard-observability-audit)
for the source and access path for each category.

## Running it

Use the maintained image where possible:

```bash
bash scripts/maintain.sh
```

For source development:

```bash
ascento dashboard start --reload
ascento dashboard status --json
```

The `dashboard start` command detaches by default. Add `--foreground` for a
terminal-owned development server. The frontend development server is separate:

```bash
cd dashboard/frontend
npm run dev
```

Vite proxies `/api` to `127.0.0.1:8000`.

## Performance and troubleshooting boundary

The backend caches run-summary scans briefly and bounds telemetry/log payloads.
Use `progress` for frequent monitoring, `telemetry` with a modest limit for
recent detail, and the chart endpoint only when the full trend is needed. A
mounted WSL workspace makes filesystem scans more expensive than an in-memory
service; low-frequency CLI/MCP monitoring is typically the lightest path.

See [Troubleshooting](troubleshooting.md) for blank metrics, a missing run,
stale status, and slow UI diagnostics.

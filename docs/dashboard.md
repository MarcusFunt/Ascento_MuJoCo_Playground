# Dashboard and HTTP API

The dashboard is a local control plane for artifact-backed RSL-RL/mjlab runs.
It monitors existing artifacts, starts and stops managed runs, compares
normalized telemetry, reports provenance and health, and can ask the guarded
host supervisor to update the checkout. The Analyze page also shows Blender
renders written under the mounted `captures/blender` directory.

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
| `GET` | `/api/health/live` | Process liveness |
| `GET` | `/api/health/ready` | Required artifact/service readiness |
| `GET` | `/api/config` | Public runtime configuration |
| `GET` | `/api/system` | Cached host/update/Tailnet status |
| `GET` | `/api/activity` | Verified trainer, evaluator, viewer, and render activity |
| `GET` | `/api/assessments` | Deterministic, read-only operational/evidence findings |
| `GET` | `/api/runtime/identity` | Checkout, packaged image, deployed API, and run revisions |
| `GET` | `/api/runtime/preflight` | Read-only task/runtime/device launch preflight |
| `GET` | `/api/control/session` | Control-session configuration and browser session state |
| `POST` | `/api/control/session` | Exchange operator token for a short-lived control session |
| `DELETE` | `/api/control/session` | Lock dashboard controls |
| `POST` | `/api/system/update` | Request guarded maintenance update |
| `GET` | `/api/experiments` | Experiment plans and explicitly linked run lineage |
| `GET` | `/api/evaluation-suites` | Discover declared suites; does not launch evaluation |
| `GET` | `/api/evaluations` | Filtered/paginated evaluation registry (`limit`, `offset`, filters; ETag revalidation) |
| `GET` | `/api/evaluations/{id}` | Evaluation detail and consistency evidence |
| `GET` | `/api/evaluations/{id}/gates` | Gate results, preserving PASS/FAIL/INCOMPLETE/INVALID |
| `GET` | `/api/evaluations/{id}/scenarios` | Bounded scenario failures/metrics |
| `GET` | `/api/evaluations/compare` | Contract-compatible evaluation comparison |
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
| `GET` | `/api/runs/{id}/checkpoints` | Published checkpoint list and stability state |
| `GET` | `/api/runs/{id}/checkpoint-compatibility` | Exact stable-checkpoint hash and task/plant/action ABI preflight |
| `GET` | `/api/runs/{id}/checkpoint-evidence` | Stable checkpoint SHA-256 and linked evidence |
| `GET` | `/api/blender/renders` | Recent Blender render manifests and safe media URLs |
| `GET` | `/api/blender/renders/files/{path}` | Serve `.blend`, `.mp4`, `.png`, or manifest files under `captures/blender` |

Evaluation and capture endpoints are read-only; they do not launch a suite,
change promotion rules, or modify artifact content. Use the CLI or MCP for
those operations so all artifact handling follows the same project contract.
Blender media routes are read-only and restricted to the mounted
`captures/blender` tree; the host-side render wrapper writes to the WSL path,
while the dashboard container keeps its capture mount read-only.

The New Run wizard lists only published stable parent checkpoints. It checks
the exact artifact against the selected task's plant, action, and task contracts
before enabling Start, and the create API repeats that validation server-side.
The request carries a checkpoint path relative to the selected parent run; the
server resolves it within that run and passes the validated absolute path to the
launcher. A missing or incompatible checkpoint blocks the launch.

### Dashboard control access

Read APIs remain available to dashboard viewers. Run start/stop and metadata
changes, host update requests, and viewer/waypoint/capture controls require an
operator session. Configure `ASCENTO_CONTROL_TOKEN` on the dashboard service
with a randomly generated secret of at least 32 characters; keep it in the
host's untracked Compose environment file or secret manager, never in Git. The
UI exchanges it for a 12-hour, HttpOnly, SameSite=Strict cookie and does not
persist the token in browser storage. Locking clears the cookie. Each mutation
also checks the request Origin and rejects `Sec-Fetch-Site: cross-site`; behind
a proxy with a different browser hostname, set
`ASCENTO_DASHBOARD_ALLOWED_ORIGINS` to a comma-separated list of exact origins
(scheme and host, no path).

The Docker service receives only the control secret and restricted supervisor
socket; it does not receive the Docker socket. Mutation events are recorded in
the dashboard activity index when PostgreSQL is available. The dashboard does
not start evaluations or held-out promotion runs from these pages.

Evaluation listing is capped at 5,000 discovered summaries and 500 rows per
page. Discovery is cached for 10 seconds; list responses include a content
ETag and support `If-None-Match`. This bounds filesystem scans while retaining
prompt revalidation. Evaluation detail and scenario APIs are loaded only when
opened and remain read-only.

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

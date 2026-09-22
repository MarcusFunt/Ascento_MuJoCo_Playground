# Dashboard control room

The Dashboard is a purpose-built training control room for Ascento mjlab/RSL-RL
work. Its default route is **Overview**: opening the site should immediately
answer which run is active, whether learning is healthy, where the policy is in
its curriculum, and what changed recently.

The interface intentionally does not use Grafana, Prometheus, Redux, or a second
design system.

## Frontend architecture

The frontend is React + TypeScript + Vite.

- **TanStack Router** owns real browser routes, including direct links to runs
  and analysis pages.
- **TanStack Query** owns asynchronous server state, caching, invalidation, and
  polling. Components do not implement their own in-flight request locks.
- **Tailwind CSS v4** provides design tokens and layout primitives.
- UI primitives follow the **shadcn open-code model** and are implemented over
  **Base UI**. The source lives in this repository.
- **TanStack Table** drives the run library.
- **Apache ECharts** powers the zoomable Analyze charts. Analyze is lazy-loaded
  so ECharts is not part of the landing-page bundle.
- **React Hook Form + Zod** validate managed-run creation.
- **cmdk** provides the Ctrl+K command palette.
- **Lucide** provides the small icon vocabulary.

The theme is deliberately near-black and mostly achromatic. Green, amber, and
red are reserved for state rather than decoration. Typography and spacing are
larger than the previous dashboard so fewer elements compete for attention.

### Routes

- `/` — Overview / control room
- `/runs` — run library and comparison
- `/runs/:runId` — one run, curriculum, provenance, metadata, and policy viewer
- `/analyze` — analysis with run selection
- `/analyze/:runId` — directly linked analysis workspace
- `/system` — repository, Tailnet, supervisor, and dashboard infrastructure

FastAPI serves the SPA shell for direct browser routes, so refreshing a run or
analysis URL does not lose the current page.

## Task-aware curriculum UI

Curriculum state is a first-class domain model, not a generic telemetry chart.

### Horizon curriculum

Balance, velocity, and balance-recovery expose the adaptive horizon schedule:

`20 s -> 60 s -> 120 s -> 300 s`

The UI shows the current stage, completed/upcoming stages, the six-window
promotion streak, timeout and quality thresholds, severe-failure streak, and
the protected final stage. The values come from the same runner status emitted
by `HorizonCurriculumRunner`.

### Balance recovery

Balance-recovery additionally shows the hard-reset mixture and the
120,000-control-step difficulty ramp, including the current pose/velocity
difficulty envelope.

### Locomotion

Locomotion shows the actual per-environment training sequence:

`settle -> push -> recover -> target -> stop`

This is descriptive rather than pretending the task has a scalar stage that it
does not emit.

## Backend APIs for the new UI

The old detailed endpoints remain compatible. The control room adds bounded,
purpose-specific APIs:

- `GET /api/overview` — one request for the active run, critical metrics,
  curriculum, bounded sparkline history, recent events, and counts.
- `GET /api/runs/index` — compact run library; no full task/action/plant
  contract payloads.
- `GET /api/tasks` — backend-owned task catalog used by the new-run form.
- `GET /api/runs/:id/curriculum` — normalized task-aware curriculum state.

Detailed provenance and compatibility contracts remain on `GET /api/runs/:id`.

## PostgreSQL semantic index

Training artifacts are still authoritative on disk. Checkpoints, TensorBoard
files, console logs, manifests, renders, and evaluation output are **not** moved
into PostgreSQL.

The Dashboard has a small SQLAlchemy/Alembic-managed PostgreSQL index for compact
run state, checkpoint metadata, curriculum snapshots, run state transitions,
and curriculum transition events.

The Compose service uses a persistent `dashboard-postgres-data` volume and is
not exposed on a host port. The Dashboard never returns the database password
through its public configuration endpoint.

`ASCENTO_DATABASE_URL` enables the index. If the database is unavailable, the
application records a startup warning and the filesystem monitor/control
features remain usable.

Schema changes are managed with Alembic:

    uv run --extra dashboard alembic upgrade head

The container runs migrations automatically during Dashboard startup.

## Data ownership

| Data | Authority |
| --- | --- |
| checkpoints / model files | filesystem |
| TensorBoard / telemetry source | filesystem |
| raw training logs | filesystem |
| run manifests and exact provenance | filesystem |
| renders / videos / evaluations | filesystem |
| compact UI index | PostgreSQL |
| transition/activity events | PostgreSQL |

The database is rebuildable and cannot make a valid training artifact
unreadable.

## UI performance rules

- Overview polling uses `/api/overview`; it must stay bounded.
- The run library uses `/api/runs/index`; it must not include large contracts.
- Heavy charts are lazy-loaded with the Analyze route.
- Overview sparklines are tiny local SVGs instead of chart-library instances.
- Archived detail and full telemetry load only after the user asks for them.
- Logs continue to stream over SSE.
- Viser remains a separate managed process and WebSocket service.

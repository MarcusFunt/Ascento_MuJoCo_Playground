# Development and validation

## Repository map

| Path | Purpose |
| --- | --- |
| `src/ascento_mjlab/assets/` | Robot MJCF asset |
| `src/ascento_mjlab/physics.py` | Canonical timing/authority profile |
| `src/ascento_mjlab/robot_cfg.py` and `actuator*.py` | Entity, action/actuator, supported pose, simulator configuration |
| `src/ascento_mjlab/mdp/` | Commands, resets/events, observations, rewards, state, metrics, termination |
| `src/ascento_mjlab/tasks/` | Registered task and PPO configurations |
| `src/ascento_mjlab/evaluation/` | Versioned scenario materialization, runner, statistics, gates, report, replay |
| `src/ascento_mjlab/tools/` | Smoke, inspection, capture, clip, quality, reward, and plant tools |
| `dashboard/` | FastAPI backend, artifact health, run service, frontend, supervisor client |
| `scripts/` | Bootstrap, preflight, dashboard, supervisor, Tailnet, MCP integration |
| `benchmarks/suites/` | Immutable evaluation suites |
| `tests_mjlab/`, `tests_dashboard/` | Python validation suites |

## Local validation

Use the same compute extra selected for the environment:

```bash
uv run --frozen --extra cpu --extra dashboard ruff check .
uv run --frozen --extra cpu --extra dashboard pytest -q
```

For plant, model, CUDA, and finite-step smoke on a production-style machine:

```bash
uv run --frozen --extra cu128 python -m ascento_mjlab.tools.smoke
uv run --frozen --extra cu128 python -m ascento_mjlab.tools.inspect_model
bash scripts/preflight_long_run.sh
```

Build frontend changes before publishing them:

```bash
cd dashboard/frontend
npm ci
npm run build
```

Check shell and Compose changes without mutating a run:

```bash
bash -n scripts/run_dashboard.sh scripts/maintain.sh scripts/install_supervisor.sh scripts/setup_tailscale.sh
docker compose -f docker/compose.yaml config
```

## Change-specific expectations

| Change | Minimum relevant verification |
| --- | --- |
| Reward, reset, termination, task, actuator, physics, or horizon code | Unit/integration tests plus the relevant deterministic suite or preflight |
| Evaluation runner/schema/gates | Evaluator tests, deterministic preflight, and a representative suite run |
| CLI, MCP, operations, or artifact code | Focused CLI/MCP/operations tests, JSON documentation check, and help smoke |
| Dashboard backend | `tests_dashboard`, frontend build if API/UI behavior changes |
| Frontend | `npm run build`; inspect trend span and blank-metric states manually when changing charts |
| Docker/maintenance/Tailscale | Shell syntax and `docker compose config`; avoid running destructive maintenance in a dirty checkout |
| Documentation or interface inventory | JSON parsing and `tests_mjlab/test_documentation_contract.py` |

## Evaluation and task evolution

Keep training terms, metrics, and gates separate. Changing a reward does not
automatically justify changing a benchmark. If the metric or accepted operating
envelope changes, add a new TOML suite instead of editing the published suite.
Record exact reward changes, seed, environment count, checkpoint, source
commit, and evaluator result in run metadata/evaluation artifacts.

When adding a task, register training and play environments in
`tasks/__init__.py`, add PPO config, task-specific MDP terms and metrics,
versioned evaluation coverage, artifacts/CLI documentation, and tests. When
adding an advanced dashboard metric, ensure the launcher/telemetry parser emits
it; a visual card alone cannot manufacture historical data.

## CI coverage

The GitHub workflow runs CPU mjlab tests, dashboard tests/lint/shell checks,
Docker/Compose validation, and frontend build checks for relevant source,
benchmark, dashboard, operations, test, documentation, and interface changes.
CI is a regression screen, not a substitute for GPU/physics preflight or an
authoritative policy evaluation.

## Keeping docs accurate

Update both the human page and machine inventory when a public operation
changes. The root [documentation index](README.md) identifies the mappings.
The project-local Codex skill is part of the interface documentation and should
be updated with significant CLI semantics, not just command spelling.

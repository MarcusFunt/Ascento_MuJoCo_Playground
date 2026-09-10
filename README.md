# Ascento MuJoCo Playground

Simulation-only motion authoring for an Ascento Guard-2-like wheel-legged
robot. The current migration target is mjlab 1.6.0, MuJoCo Warp, and RSL-RL;
the project is not intended for physical-robot deployment.

## Documentation

The root README is the short operational entry point. The maintained
documentation set is indexed in [docs/README.md](docs/README.md):

- [architecture](docs/architecture.md) and the simulation/plant contract;
- [training](docs/training.md), [evaluation](docs/evaluation.md), and the
  task/gate workflow;
- [operations](docs/operations.md), [dashboard](docs/dashboard.md), and
  [troubleshooting](docs/troubleshooting.md);
- the complete [CLI reference](docs/cli-reference.md) and
  [MCP reference](docs/mcp-reference.md);
- machine-readable inventories in
  [docs/project-manifest.json](docs/project-manifest.json),
  [docs/cli-reference.json](docs/cli-reference.json), and
  [docs/mcp-tools.json](docs/mcp-tools.json).

The project-local Codex skill at
[.codex/skills/ascento-cli/SKILL.md](.codex/skills/ascento-cli/SKILL.md) is the
agent-oriented guide for using the same CLI safely.

## Architecture

```text
MJCF → MuJoCo Warp → mjlab managers → RSL-RL PPO → RecorderManager → animation export
```

mjlab owns scene/entity construction, action and observation managers, commands,
events, resets, rewards, terminations, metrics, recording, and vectorized
lifecycle. RSL-RL owns PPO. This repository owns the robot MJCF, the small
Ascento actuator extension, motion-specific MDP terms, task configs, capture,
and tests.

The `structured_targets_v1` controller uses six normalized actions in `[-1, 1]`:
four leg-position targets around the nominal pose and two wheel-velocity targets.
Physics-rate leg PD and wheel PI controllers request torque before the existing
65 Nm motor envelope, controller-speed protection, and finite response model.
The 15 Nm leg and 5 Nm wheel continuous ratings are documentation only until a
thermal/duration model is justified. No communication delay, sensor noise, or
thermal model is enabled.

## Maintenance / first-time install

For a normal Linux/WSL2 machine, the preferred setup and update path is the
maintenance script:

```bash
bash scripts/maintain.sh
```

When run from an existing checkout it updates that checkout. The same script can
also be downloaded and run on a machine that has never cloned the repository;
it defaults to `~/Ascento_MuJoCo_Playground`:

```bash
curl -fsSL https://raw.githubusercontent.com/MarcusFunt/Ascento_MuJoCo_Playground/main/scripts/maintain.sh | bash
```

The maintainer:

- installs missing base tooling, Docker Engine/Compose, uv, and NVIDIA Container
  Toolkit where appropriate on supported Debian/Ubuntu systems;
- automatically chooses CUDA when a usable NVIDIA GPU is present, otherwise CPU;
- updates the checkout to the requested remote branch without deleting
  `logs/`, `checkpoints/`, or `captures/`;
- runs exact `uv sync`, including every dependency group and every optional
  dependency compatible with the selected compute backend;
- uses `npm ci` for an exact frontend dependency reconciliation and rebuilds the
  dashboard;
- rebuilds the Docker image from scratch so removed image dependencies cannot
  remain in the active image;
- starts the dashboard container on `127.0.0.1:8000` by default;
- preserves existing runs and records pre-update Git provenance for legacy runs
  that did not already record their repository commit;
- preserves the optional Tailscale dashboard ingress after it has been enrolled.

`cpu` and `cu128` are declared as conflicting extras, so only one can be
installed at a time. All other compatible optional extras are installed.
`uv sync` is exact by default, so Python packages that disappear from the
lockfile/project are removed. `npm ci` likewise replaces `node_modules` with the
contents of the current lockfile.

The script refuses to overwrite tracked local changes or local-only commits
unless `--force` is explicitly supplied. See `bash scripts/maintain.sh --help`
for CPU/GPU, install-directory, and Docker options.

### Dashboard host updates

The Dashboard's **System** page can compare the local checkout with
`origin/main` and request an update, but the web container never receives the
Docker socket or arbitrary host shell access. Install the small host-side
supervisor once:

```bash
bash scripts/install_supervisor.sh
```

The supervisor runs as the workstation user under systemd and exposes only a
restricted Unix socket to the Dashboard. It accepts repository-status checks and
a guarded request to run the existing `scripts/maintain.sh`; it does not accept
arbitrary commands, branches, paths, or Docker operations from the web API.

GUI-triggered updates are refused while training is active, while the checkout
is dirty, when local-only commits exist, when the checkout is not on `main`, or
when `origin/main` cannot be verified. When an update is accepted, the supervisor
continues running outside Docker while the Dashboard is rebuilt and restarted.

### Remote Dashboard over Tailscale

The maintained Docker stack can expose the Dashboard privately to your Tailscale
tailnet without opening it to the LAN or public internet. Enroll it once on the
workstation:

```bash
bash scripts/setup_tailscale.sh
```

Supply a Tailscale auth key at the hidden prompt or through `TS_AUTHKEY`. The
credential is used for initial enrollment only; after persistent Tailscale state
has been created, the sidecar is recreated without the credential in its Docker
environment. For OAuth client-secret enrollment, also provide the required
advertised tag, for example:

```bash
ASCENTO_TAILSCALE_EXTRA_ARGS='--advertise-tags=tag:ascento' \
TS_AUTHKEY='<oauth-client-secret>' \
bash scripts/setup_tailscale.sh
```

The Tailscale sidecar owns the Dashboard's network namespace. Tailnet peers can
open port `8000` on its Tailscale IP or MagicDNS name, while local access remains
available on `127.0.0.1:8000`. The separate `ascento-mjlab` service is not joined
to that Tailnet ingress. Tailscale Funnel is not enabled; your Tailnet grants or
ACLs determine who can reach the private Dashboard.

## Manual setup

Use Linux/WSL2, Python 3.11–3.13, an NVIDIA driver compatible with the pinned
CUDA wheel, and `uv`:

```bash
uv sync --extra cu128 --extra dashboard
```

For CPU-only development:

```bash
uv sync --extra cpu --extra dashboard
```

The lockfile pins mjlab 1.6.0, MuJoCo 3.11, MuJoCo Warp, Warp, Torch, and
RSL-RL. When using the CUDA environment, keep `--extra cu128` on `uv run`
commands or use the environment created by `uv sync --extra cu128`.

## Operations CLI and MCP

The unified `ascento` CLI uses the same run service as the dashboard:

```bash
uv run --extra dashboard ascento run start --task Ascento-Balance-Flat \
  --display-name "Balance baseline" --envs 512 --iterations 10000 --seed 123
uv run --extra dashboard ascento run list --active
uv run --extra dashboard ascento run monitor <run-id> --interval 30
uv run --extra dashboard ascento run progress <run-id> --json
uv run --extra dashboard ascento run logs <run-id> --tail 100
uv run --extra dashboard ascento run telemetry <run-id> --limit 50 --json
uv run --extra dashboard ascento run compare <run-a> <run-b> --json
uv run --extra dashboard ascento run annotate <run-id> --tag candidate --purpose validation
uv run --extra dashboard ascento run stop <run-id>
uv run --extra dashboard ascento dashboard status
uv run ascento maintain --skip-system-install
```

Arguments after `--` are passed directly to the mjlab trainer. For agent
workflows, the unified CLI also exposes the evaluator, report artifacts, policy
clips, and specialist tools:

```bash
uv run --extra cu128 --extra dashboard ascento evaluate suites
uv run --extra cu128 --extra dashboard ascento evaluate run \
  --run-id <run-id> --suite balance_gate_v3 --render-clips
uv run --extra dashboard ascento evaluate list
uv run --extra dashboard ascento evaluate report <evaluation-id>
uv run --extra dashboard ascento evaluate archive <evaluation-id>
uv run --extra cu128 --extra dashboard ascento capture \
  --run-id <run-id> --takes 3 --steps 900 --video-dir captures/balance/videos
uv run --extra cu128 ascento tools clip-motion -- captures/balance/take_000.npz --fps 24
uv run --extra cu128 ascento tools rank-motion -- captures/balance --top 5
```

`evaluate run` accepts either `--checkpoint` or `--run-id`; its report contains
the immutable suite snapshot, gates, consistency checks, failure modes, SQLite
results, and optional capture/video manifest. `evaluate archive` creates a ZIP
of that complete evidence directory. `capture` infers the registered task when
given a managed run ID.

The stdio server can still be launched explicitly for a generic MCP client:

```bash
uv run --extra dashboard --extra mcp ascento-mcp
# equivalent: uv run --extra dashboard --extra mcp ascento mcp serve
```

For Codex Desktop on this Windows/WSL workstation, register the project server
once from PowerShell:

```powershell
.\scripts\register_codex_mcp.ps1
```

That configures Codex to start `scripts/ascento_mcp_server.sh` inside the
`Ubuntu` WSL distribution, so it uses the project's locked Linux runtime rather
than a separate host Python environment. Use `-Distro <name>` for a different
distribution, and `-Replace` only when intentionally replacing an existing
`ascento` MCP registration. Open a new Codex Desktop task (or restart the app)
after registration; a running task cannot acquire new tools mid-session.

The MCP server exposes run listing/progress/details/logs/telemetry, run
start/stop/metadata updates/comparison, checkpoint resolution, immutable suite
listing, full evaluation, report inspection/comparison/ZIP export, policy
capture, and deterministic evaluator preflight. Set `ASCENTO_ARTIFACT_ROOT`
and `ASCENTO_EVALUATION_ROOT` when those artifacts live outside the default
`logs/rsl_rl` and `evaluations` directories.

## Validate the plant

```bash
uv run --extra cu128 python -m ascento_mjlab.tools.smoke
uv run --extra cu128 python -m ascento_mjlab.tools.inspect_model
uv run --extra cu128 --extra dashboard pytest -q
```

Immediately before committing to a long run, use the production host and the
actual balance, velocity, and recovery checkpoints to run the complete preflight:

```bash
export ASCENTO_BALANCE_CHECKPOINT=logs/rsl_rl/ascento_balance/<run>/model_<n>.pt
export ASCENTO_VELOCITY_CHECKPOINT=logs/rsl_rl/ascento_velocity/<run>/model_<n>.pt
export ASCENTO_RECOVERY_CHECKPOINT=logs/rsl_rl/ascento_recovery/<run>/model_<n>.pt
export ASCENTO_COMPUTE_EXTRA=cu128
bash scripts/preflight_long_run.sh
```

That command runs the finite plant smoke test, model inspection, the full test
suite, a two-iteration 512-environment PPO smoke using the production training
path, and deterministic quantitative evaluator preflight. The evaluator runs the
same scenarios twice and fails on non-finite outputs, ambiguous episode endings,
unfinished vector slots, mixed-horizon lifecycle errors, or deterministic result
drift. Do not launch a long run if this command fails.

The smoke command prints Torch/CUDA/GPU/VRAM, MuJoCo, Warp, mjlab, and
RSL-RL versions, then runs 100 finite Warp steps. Start balance training with
the standard mjlab entry point after Gate D review:

```bash
uv run --extra cu128 train Ascento-Balance-Flat --env.scene.num-envs 512
uv run --extra cu128 play Ascento-Balance-Flat --agent zero
```

Gate D is mjlab-native: the validated plant must learn robust, visually
plausible balance with sensible control. Balance rewards hold the root near its
supported reset position, reward the same settled state measured by the gate,
and softly prefer equal mirrored hip/knee coordinates without coupling their
actions. Training applies one cardinal planar recovery push per episode between
4 and 6 seconds, matching the gate's disturbance envelope; exact evaluator
pushes are still provided only by the immutable scenario suite. The balance and
velocity curriculum progresses through 20, 60, 120, and 300 seconds. It protects
the final 300-second phase from stochastic training-rollout demotions and writes
`model_best_long_horizon.pt` after sustained top-stage survival. That file is a
candidate, not a passing result: screen it and regular checkpoints with the
deterministic `balance_gate_v3` suite before selecting a model. Pre-contract policy
behavior is a diagnostic reference only, never the acceptance target.

## Tasks and sequencing

Registered tasks are:

- `Ascento-Balance-Flat`
- `Ascento-Velocity-Flat`
- `Ascento-Recovery-Flat`
- `Ascento-Jump-Flat`

The flat-ground order is balance, velocity/yaw/height, recovery, then jump.
Jump state is derived from wheel contact and root motion until evidence requires
one persistent state owner. One request must produce one attempt; no parallel
FSM or generalized scenario DSL is used.

Terrain is a hard sequencing gate: raised surfaces, slopes, obstacles,
clearance, and high-landing variants are not expanded until flat-ground jump
takeoff, flight, landing, and post-landing recovery are demonstrably sound.

## Capture

Capture uses mjlab's RecorderManager and emits named state channels suitable for
downstream animation tooling:

```bash
uv run --extra cu128 python -m ascento_mjlab.tools.capture_motion \
  --task Ascento-Jump-Flat --checkpoint logs/rsl_rl/ascento_jump/model_10000.pt \
  --takes 20 --steps 1000 --output-dir captures/jump
```

Omit `--checkpoint` for a zero-action plant capture. Each take includes time,
root transforms, local joint state, contact state, jump state when available,
applied effort, action, task, seed, checkpoint hash, physics profile, and
capture FPS. Trim and resample a take for animation use:

```bash
uv run clip-motion captures/jump/take_000.npz \
  --event takeoff --pre-roll 0.5 --post-roll 1.0 --fps 24 \
  --output captures/jump/jump_short.npz
```

Rank a batch of captures by smoothness and contact quality. This quality score
is separate from task-success acceptance and is intended to select candidates
for visual review:

```bash
uv run rank-motion captures/jump --top 5 --output captures/jump/ranking.json
```

## Docker

The maintained image includes the Python project, all compatible optional
extras/dependency groups, dashboard backend, and a freshly built frontend.
The base compose file is CPU-safe; add the GPU overlay for CUDA:

```bash
# CPU
docker compose -f docker/compose.yaml build

# NVIDIA GPU
docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml build
```

The optional Tailnet overlay is normally managed by `scripts/setup_tailscale.sh`:

```bash
docker compose \
  -f docker/compose.yaml \
  -f docker/compose.tailscale.yaml config
```

The maintenance script normally handles the build arguments and starts the
`dashboard` service automatically. Training logs, checkpoints, and captures are
bind-mounted from the repository, so container rebuilds do not delete runs.
The Dashboard has read-write access to `logs/` because managed runs create their
status, TensorBoard output, logs, and checkpoints there; separate
`checkpoints/`, `captures/`, and `evaluations/` mounts stay read-only. The only
host-control mount is the restricted supervisor Unix socket directory; Docker's
control socket is never mounted into the Dashboard.

The dashboard compares each run's recorded Git commit with the repository
version serving the UI. Runs from older commits are labelled `OUTDATED` in the
run selector and show an explicit warning plus both commits in Run Information.
Legacy runs without Git metadata receive an inferred provenance sidecar during
the first maintenance update; the UI makes that inference visible.

## Plant comparison policy

`ascento_mjlab.tools.compare_plant` is a migration diagnostic for controlled
ordinary-MuJoCo-versus-Warp trajectories. It is not a permanent CI requirement
or a second runtime backend. Archive it after Gates D/E unless it catches a
specific important plant regression.

## Repository layout

- `src/ascento_mjlab/assets/ascento_guard2/robot.xml`: backend-neutral robot MJCF.
- `src/ascento_mjlab/robot_cfg.py`: named entity, contacts, limits, and scene constants.
- `src/ascento_mjlab/actuator.py`: peak/speed/response actuator extension.
- `src/ascento_mjlab/mdp/`: Ascento observations, rewards, resets, semantics, metrics.
- `src/ascento_mjlab/tasks/`: balance, velocity, recovery, and flat-jump configs.
- `src/ascento_mjlab/tools/`: smoke, model inspection, plant comparison, capture, clip processing, and motion-quality ranking.
- `dashboard/`: run-management, monitoring, System/update UI, and host-supervisor client.
- `scripts/host_supervisor.py`: fixed-operation host update boundary.
- `tests_mjlab/`: migration unit and integration tests.
- `tests_dashboard/`: dashboard, run-control, supervisor, and update API tests.

The old JAX/MJX/Brax implementation remains available only in Git history and
the `archive/mjx-playground` branch during migration validation.

# Ascento MuJoCo Playground

Simulation-only motion authoring for an Ascento Guard-2-like wheel-legged
robot. The current migration target is mjlab 1.6.0, MuJoCo Warp, and RSL-RL;
the project is not intended for physical-robot deployment.

## Architecture

```text
MJCF → MuJoCo Warp → mjlab managers → RSL-RL PPO → RecorderManager → animation export
```

mjlab owns scene/entity construction, action and observation managers, commands,
events, resets, rewards, terminations, metrics, recording, and vectorized
lifecycle. RSL-RL owns PPO. This repository owns the robot MJCF, the small
Ascento actuator extension, motion-specific MDP terms, task configs, capture,
and tests.

The initial direct-effort actuator uses six normalized actions in `[-1, 1]`,
40 Nm peak simulation authority, a linear torque-speed envelope, controller
speed protection, and finite torque response. Training clips the policy action
to `[-1, 1]` before the environment maps it to a physical `[-40, 40] Nm`
effort target. The 15 Nm leg and 5 Nm wheel continuous ratings are documentation
only until a thermal/duration model is justified. No communication delay, sensor
noise, or thermal model is enabled.

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
  that did not already record their repository commit.

`cpu` and `cu128` are declared as conflicting extras, so only one can be
installed at a time. All other compatible optional extras are installed.
`uv sync` is exact by default, so Python packages that disappear from the
lockfile/project are removed. `npm ci` likewise replaces `node_modules` with the
contents of the current lockfile.

The script refuses to overwrite tracked local changes or local-only commits
unless `--force` is explicitly supplied. See `bash scripts/maintain.sh --help`
for CPU/GPU, install-directory, and Docker options.

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

## Validate the plant

```bash
uv run --extra cu128 python -m ascento_mjlab.tools.smoke
uv run --extra cu128 python -m ascento_mjlab.tools.inspect_model
uv run --extra cu128 pytest -q
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
plausible balance with sensible control. The balance objective mildly penalizes
horizontal root speed so recovery motion remains available without rewarding
persistent drift. Old-stack policy behavior is a diagnostic reference only,
never the acceptance target.

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

The maintenance script normally handles the build arguments and starts the
`dashboard` service automatically. Training logs, checkpoints, and captures are
bind-mounted from the repository, so container rebuilds do not delete runs.
The dashboard mount is read-only and defaults to the shared `logs/rsl_rl`
artifact root.

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
- `tests_mjlab/`: migration unit and integration tests.

The old JAX/MJX/Brax implementation remains available only in Git history and
the `archive/mjx-playground` branch during migration validation.

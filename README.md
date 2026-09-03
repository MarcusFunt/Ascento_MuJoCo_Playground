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
speed protection, and finite torque response. The 15 Nm leg and 5 Nm wheel
continuous ratings are documentation only until a thermal/duration model is
justified. No communication delay, sensor noise, or thermal model is enabled.

## Setup

Use Linux/WSL2, Python 3.11–3.13, an NVIDIA driver compatible with the pinned
CUDA wheel, and `uv`:

```bash
uv sync --extra cu128
```

For CPU-only development:

```bash
uv sync --extra cpu
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

The smoke command prints Torch/CUDA/GPU/VRAM, MuJoCo, Warp, mjlab, and
RSL-RL versions, then runs 100 finite Warp steps. Start balance training with
the standard mjlab entry point after Gate D review:

```bash
uv run --extra cu128 train Ascento-Balance-Flat --num-envs 512
uv run --extra cu128 play Ascento-Balance-Flat --agent zero
```

Gate D is mjlab-native: the validated plant must learn robust, visually
plausible balance with sensible control. Old-stack policy behavior is a
diagnostic reference only, never the acceptance target.

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
root transforms, joint state, applied effort, action, task, seed, and capture
FPS.

## Docker

The final image uses the lockfile and NVIDIA Container Toolkit runtime:

```bash
docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml run --rm ascento-mjlab
```

Logs, checkpoints, and captures are mounted from the repository. The image
defaults to the version/GPU/finite-rollout smoke gate and supports headless
offscreen rendering through `MUJOCO_GL=egl`.

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
- `src/ascento_mjlab/tools/`: smoke, model inspection, plant comparison, capture.
- `tests_mjlab/`: migration unit and integration tests.

The old JAX/MJX/Brax implementation remains available only in Git history and
the `archive/mjx-playground` branch during migration validation.

# Ascento MuJoCo Playground

Simulation-only direct-torque motion control for an Ascento Guard 2.0-like
wheel-legged robot. The simulator is an authoring source for a 3D-to-2D sprite
pipeline; it is not intended for physical-robot deployment.

## Control architecture

```
PPO policy
  -> 6 normalized actions
  -> requested motor torques
  -> torque, speed, and bandwidth envelope
  -> MuJoCo MJX
  -> robot dynamics
```

There are no PD targets, LQR, VMC, wheel-velocity PI loops, joint-position
targets, or joint-velocity targets between the learned policy and the motors.

## Implemented task framework

Every stage uses the same six actions and the same 49-value observation. The
observation reserves forward-velocity, yaw-rate, height, and jump commands;
wheel contact/force, applied torque, prior action, and six jump-phase channels
are present from the first balance stage. Checkpoints therefore transfer without
reshaping the actor or critic.

Implemented environments:

- `AscentoBalance`: near-upright balance and command tracking;
- `AscentoRecovery`: broad but mechanically reachable tilted/velocity resets;
- `AscentoJump`: an event-tracked six-phase jump task (`IDLE`, `CROUCH`,
  `THRUST`, `FLIGHT`, `LANDING`, `RECOVERY`).

The phase tracker supplies observation, reward gating, and metrics only. It
does not generate motor commands. The policy remains the sole controller.

The staged sequence is `balance → flat_commands → recovery → jump_flat →
high_landing → clearance → moving_jump → unified_fine_tune`. Advancement uses
deterministic physical metrics, not a single total-reward value. The framework
is implemented, but each stage still needs its own training run and acceptance
results before it can be called a learned capability.

The project also includes deterministic evaluation, NPZ trajectory export, and
an open-loop direct-torque jump feasibility sweep. Blender/rendering integration
is deliberately not part of this implementation.

## Setup

Requirements: Linux or WSL2, Python 3.10–3.13, an NVIDIA driver compatible
with CUDA 12, and a CUDA-capable GPU. The dependency versions are pinned in
[`requirements-cuda.txt`](requirements-cuda.txt).

```bash
git clone https://github.com/MarcusFunt/Ascento_MuJoCo_Playground.git
cd Ascento_MuJoCo_Playground
./scripts/setup.sh --dev
source .venv/bin/activate
```

The setup script creates `.venv`, installs the pinned CUDA 12 stack, and
verifies JAX, MuJoCo, MJX, and Brax imports. It prints the selected JAX device;
stop there if it reports CPU when GPU training is expected.

## Useful commands

```bash
# Lightweight actuator model check
python test_guard2_physics.py

# Compile and smoke-test the MuJoCo model
python verify_guard2_patch.py

# Run the full implementation test suite
ASCENTO_JAX_PLATFORM=cpu pytest -q tests

# Train a direct-torque stage (use --smoke for a short validation run)
ASCENTO_JAX_PLATFORM=cuda python -m training.train \
  --stage balance --output training/artifacts/balance

# Train sequential stages; later stages transfer the previous policy only
ASCENTO_JAX_PLATFORM=cuda python -m training.staged_train

# Evaluate an artifact deterministically on fixed seeds
ASCENTO_JAX_PLATFORM=cuda python -m evaluation.evaluate \
  --stage balance --artifact training/artifacts/balance

# Check whether the physical plant can leave the ground before PPO jump training
python tools/jump_open_loop_test.py

# Export a deterministic trajectory for the future animation pipeline
ASCENTO_JAX_PLATFORM=cuda python tools/export_motion.py \
  --stage balance --artifact training/artifacts/balance \
  --output training/artifacts/balance/motion.npz

# Render deterministic MuJoCo rollouts without Blender
MUJOCO_GL=egl ASCENTO_JAX_PLATFORM=cuda python tools/render_rollout.py \
  --stage balance --artifact training/artifacts/balance \
  --output-dir tools/rendered_rollouts
```

Generated environments, videos, policy exports, and training artifacts are
intentionally ignored by Git. Keep a run manifest and evaluation results with
any artifact you decide to publish separately.

## Repository layout

- `ascento/`: canonical MJX environments, actuator, commands, observations,
  rewards, jump state machine, and curriculum definitions.
- `model/`: static Guard 2.0-like MJCF used by MJX.
- `training/`: stage-configurable Brax PPO entry point and orchestrator.
- `evaluation/`: deterministic physical-metric benchmarks.
- `tools/`: non-rendering motion export and jump feasibility diagnostics.
- `tests/`: model, JIT, actuator, contact, reward, and jump-state checks.
- `mujoco_playground/`: vendored MuJoCo Playground source used by this project.

The root-level Torch/PD tuning scripts are historical experiments. They are not
part of the canonical direct-torque training or inference path.

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

## Current scope

The tracked implementation provides the MJX physics model, direct-torque
actuator model, and a nominal upright-balance PPO smoke task. Recovery,
commanded locomotion, crouching, jumping, flight control, landing, and the
sprite-export workflow are planned but not implemented yet. See
[`PROJECT_GOALS_AND_ASSUMPTIONS.md`](PROJECT_GOALS_AND_ASSUMPTIONS.md) for the
full target and [`PHYSICS_CHANGES.md`](PHYSICS_CHANGES.md) for physics choices.

## Setup

Requirements: Linux or WSL2, Python 3.10–3.13, an NVIDIA driver compatible
with CUDA 12, and a CUDA-capable GPU. The dependency versions are pinned in
[`requirements-cuda.txt`](requirements-cuda.txt).

```bash
git clone <your-repository-url>
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

# Train the current nominal-balance smoke task
ASCENTO_JAX_PLATFORM=cuda python -m training.train

# Evaluate a saved PPO artifact and optionally render an MP4
ASCENTO_JAX_PLATFORM=cuda python evaluation/benchmark_balance.py \
  --artifact training/artifacts_cuda_smoke \
  --save-mp4 training/artifacts_cuda_smoke/best_balance.mp4
```

Generated environments, videos, policy exports, and training artifacts are
intentionally ignored by Git. Keep a run manifest and evaluation results with
any artifact you decide to publish separately.

## Repository layout

- `ascento/`: canonical MJX environment and actuator implementation.
- `model/`: static Guard 2.0-like MJCF used by MJX.
- `training/`: current PPO training entry point.
- `evaluation/` and `verification/`: rollout and actuator/model checks.
- `mujoco_playground/`: vendored MuJoCo Playground source used by this project.

The root-level Torch/PD tuning scripts are historical experiments. They are not
part of the canonical direct-torque training or inference path.

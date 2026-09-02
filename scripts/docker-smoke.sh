#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE=(docker compose)
RUN_NAME="${SMOKE_RUN_NAME:-docker_smoke_balance}"
if [[ -e "docker-data/artifacts/runs/$RUN_NAME" ]]; then
  RUN_NAME="${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)"
fi

echo "== GPU passthrough =="
docker run --rm --gpus all ubuntu nvidia-smi

echo "== Build development image =="
"${COMPOSE[@]}" build dev

echo "== JAX GPU backend =="
"${COMPOSE[@]}" run --rm dev \
  python -c 'import jax; print(jax.default_backend()); print(jax.devices())'

echo "== MuJoCo/MJX imports =="
"${COMPOSE[@]}" run --rm dev \
  python -c 'import mujoco; from mujoco import mjx; print(mujoco.__version__)'

echo "== CPU tests =="
"${COMPOSE[@]}" run --rm \
  -e ASCENTO_JAX_PLATFORM=cpu \
  -e JAX_PLATFORMS=cpu \
  dev pytest -q tests

echo "== Model checks =="
"${COMPOSE[@]}" run --rm dev python test_guard2_physics.py
"${COMPOSE[@]}" run --rm dev python verify_guard2_patch.py

echo "== GPU smoke training: $RUN_NAME =="
"${COMPOSE[@]}" run --rm trainer \
  python -m dashboard.launch \
  --artifact-root /artifacts/runs \
  --name "$RUN_NAME" \
  -- \
  --stage balance \
  --smoke \
  --timesteps "${SMOKE_TIMESTEPS:-100000}" \
  --num-envs "${SMOKE_NUM_ENVS:-64}"

echo "== Headless render =="
"${COMPOSE[@]}" run --rm dev \
  python -m dashboard.render_latest \
  --stage balance \
  --checkpoint-dir "/artifacts/runs/$RUN_NAME/checkpoint" \
  --output-dir "/artifacts/runs/$RUN_NAME/renders"

echo "Smoke run completed: docker-data/artifacts/runs/$RUN_NAME"

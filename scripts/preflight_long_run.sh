#!/usr/bin/env bash
set -euo pipefail

EXTRA="${ASCENTO_COMPUTE_EXTRA:-cu128}"
PY=(uv run --frozen --extra "$EXTRA")

"${PY[@]}" python -m ascento_mjlab.tools.smoke
"${PY[@]}" python -m ascento_mjlab.tools.inspect_model
"${PY[@]}" python -m pytest -q

# Two PPO iterations at the production vector width catch runner/optimizer/GPU
# failures without pretending that the resulting policy is useful.
"${PY[@]}" train Ascento-Balance-Flat \
  --env.scene.num-envs 512 \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.run-name preflight-smoke

if [[ -n "${ASCENTO_BALANCE_CHECKPOINT:-}" \
   && -n "${ASCENTO_VELOCITY_CHECKPOINT:-}" \
   && -n "${ASCENTO_RECOVERY_CHECKPOINT:-}" ]]; then
  "${PY[@]}" ascento-preflight-evaluator \
    --balance-checkpoint "$ASCENTO_BALANCE_CHECKPOINT" \
    --velocity-checkpoint "$ASCENTO_VELOCITY_CHECKPOINT" \
    --recovery-checkpoint "$ASCENTO_RECOVERY_CHECKPOINT" \
    --device auto
else
  echo "Evaluator preflight skipped: set ASCENTO_{BALANCE,VELOCITY,RECOVERY}_CHECKPOINT." >&2
  exit 3
fi

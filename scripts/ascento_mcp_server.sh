#!/usr/bin/env bash
# Launch the project stdio MCP server from an arbitrary working directory.
# Keep stdout reserved for MCP JSON-RPC; diagnostics belong on stderr.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' \
    'Ascento MCP startup failed: the project virtual environment is missing. Run scripts/maintain.sh first.' \
    >&2
  exit 127
fi

export ASCENTO_ARTIFACT_ROOT="${ASCENTO_ARTIFACT_ROOT:-$REPO_ROOT/logs/rsl_rl}"
exec "$PYTHON" -m ascento_mjlab.mcp_server

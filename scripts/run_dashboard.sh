#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dashboard/frontend/dist"

if [[ ! -f "$DIST/index.html" ]]; then
  echo "Dashboard frontend is not built." >&2
  echo "Run: cd dashboard/frontend && npm install && npm run build" >&2
  exit 1
fi

cd "$ROOT"
exec python -m uvicorn dashboard.app:app --host 127.0.0.1 --port "${ASCENTO_DASHBOARD_PORT:-8000}"

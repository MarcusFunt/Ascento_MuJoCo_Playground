#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAINTENANCE="$REPO_ROOT/.maintenance"
ENVFILE="$MAINTENANCE/compose.env"
MARKER="$MAINTENANCE/tailscale-enabled"

usage() {
  cat <<'EOF'
Usage: scripts/setup_tailscale.sh

Enroll the Dashboard ingress sidecar in your Tailscale tailnet. Supply the auth
credential through TS_AUTHKEY or enter it at the hidden prompt. The credential
is used only for initial enrollment and is removed from the recreated Docker
container immediately after state has been persisted in the named volume.

Optional environment variables:
  ASCENTO_TAILSCALE_HOSTNAME     Tailnet hostname (default: ascento-dashboard)
  ASCENTO_TAILSCALE_EXTRA_ARGS   Additional tailscaled login args. Required for
                                 OAuth client secrets, e.g.
                                 --advertise-tags=tag:ascento
  TS_AUTHKEY                     Tailscale auth key or OAuth client secret
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }
[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: Tailscale Docker sidecar requires Linux/WSL2" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: Docker Compose plugin is required" >&2; exit 1; }
[[ -f "$ENVFILE" ]] || {
  echo "ERROR: $ENVFILE does not exist. Run scripts/maintain.sh once first." >&2
  exit 1
}
[[ -e /dev/net/tun ]] || {
  echo "ERROR: /dev/net/tun is unavailable. Tailscale kernel networking cannot start." >&2
  exit 1
}

ACTIVE_RUNS="$(python3 - "$REPO_ROOT/logs/rsl_rl" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
active = []
if root.is_dir():
    for path in root.rglob('run_status.json'):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if value.get('state') in {'starting', 'running', 'stopping'}:
            active.append(str(path.parent.relative_to(root)))
print('\n'.join(active))
PY
)"
if [[ -n "$ACTIVE_RUNS" ]]; then
  echo "ERROR: Tailnet enrollment recreates the Dashboard and is blocked while runs are active:" >&2
  printf '  %s\n' "$ACTIVE_RUNS" >&2
  exit 1
fi

AUTH_KEY="${TS_AUTHKEY:-}"
if [[ -z "$AUTH_KEY" ]]; then
  read -rsp "Tailscale auth key / OAuth client secret: " AUTH_KEY
  echo
fi
[[ -n "$AUTH_KEY" ]] || { echo "ERROR: no Tailscale credential supplied" >&2; exit 1; }

COMPUTE="$(awk -F= '$1 == "ASCENTO_COMPUTE_EXTRA" {print $2}' "$ENVFILE" | tail -n1)"
BASE_FILES=(-f "$REPO_ROOT/docker/compose.yaml")
[[ "$COMPUTE" == "cu128" ]] && BASE_FILES+=(-f "$REPO_ROOT/docker/compose.gpu.yaml")
FILES=("${BASE_FILES[@]}" -f "$REPO_ROOT/docker/compose.tailscale.yaml")

compose() {
  docker compose --env-file "$ENVFILE" "${FILES[@]}" "$@"
}

tailscale_running() {
  compose exec -T tailscale-dashboard tailscale status --json 2>/dev/null \
    | python3 "$REPO_ROOT/scripts/tailscale_status.py" >/dev/null 2>&1
}

wait_for_tailscale_running() {
  local phase="$1"
  local attempts="${2:-120}"
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if tailscale_running; then
      return 0
    fi
    sleep 0.5
  done
  echo "ERROR: Tailscale did not reach authenticated Running state during $phase." >&2
  return 1
}

mkdir -p "$MAINTENANCE"
echo "Enrolling ${ASCENTO_TAILSCALE_HOSTNAME:-ascento-dashboard} in the tailnet..."
# Release the local 127.0.0.1 dashboard port before the sidecar takes ownership
# of that same port in the shared network namespace.
docker compose --env-file "$ENVFILE" "${BASE_FILES[@]}" stop dashboard >/dev/null 2>&1 || true
TS_AUTHKEY="$AUTH_KEY" compose up -d tailscale-dashboard dashboard

# Do not treat a successful `tailscale status --json` invocation as readiness by
# itself. The CLI can emit valid JSON while containerboot is still in NeedsLogin
# or Starting. Removing TS_AUTHKEY at that point races authentication and can
# leave the recreated sidecar repeatedly asking for interactive browser login.
if ! wait_for_tailscale_running "initial enrollment"; then
  compose logs --tail=120 tailscale-dashboard >&2 || true
  exit 1
fi

# Recreate without TS_AUTHKEY so the secret is no longer present in Docker's
# inspectable container environment. Persistent Tailscale state keeps the node
# enrolled and TS_AUTH_ONCE prevents unnecessary reauthentication.
unset TS_AUTHKEY
AUTH_KEY=""
compose up -d --force-recreate tailscale-dashboard dashboard

if ! wait_for_tailscale_running "credential-free reconnect"; then
  compose logs --tail=120 tailscale-dashboard >&2 || true
  exit 1
fi

printf 'enabled_at=%s\nhostname=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${ASCENTO_TAILSCALE_HOSTNAME:-ascento-dashboard}" >"$MARKER"
chmod 600 "$MARKER"

STATUS_JSON="$(compose exec -T tailscale-dashboard tailscale status --json)"
DNS_NAME="$(printf '%s' "$STATUS_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("Self") or {}).get("DNSName", "").rstrip("."))')"
DASHBOARD_PORT="$(awk -F= '$1 == "ASCENTO_DASHBOARD_PORT" {print $2}' "$ENVFILE" | tail -n1)"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"

echo "Tailnet enrollment complete."
if [[ -n "$DNS_NAME" ]]; then
  echo "Remote Dashboard: http://$DNS_NAME:$DASHBOARD_PORT"
else
  echo "Use the Tailscale IP shown by the tailscale-dashboard service."
fi
echo "Local Dashboard:  http://127.0.0.1:$DASHBOARD_PORT"
echo "Only the Dashboard service identity is exposed to the tailnet; the ascento-mjlab service remains private."

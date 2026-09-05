#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOCKET_DIR="$REPO_ROOT/.maintenance/supervisor"
SOCKET_PATH="$SOCKET_DIR/supervisor.sock"
PYTHON_BIN="${ASCENTO_SUPERVISOR_PYTHON:-$(command -v python3 || true)}"
SERVICE_NAME="ascento-supervisor.service"

usage() {
  cat <<EOF
Usage: scripts/install_supervisor.sh

Install the Ascento host supervisor as a boot-persistent systemd service that
runs as the current user. The service exposes only its Unix socket to the
Dashboard container; Docker's control socket is never mounted into the Dashboard.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }

[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: supervisor installation requires Linux/WSL2" >&2; exit 1; }
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python3 is required" >&2; exit 1; }
mkdir -p "$SOCKET_DIR"
chmod 700 "$SOCKET_DIR"

if ! command -v systemctl >/dev/null 2>&1 || [[ ! -d /run/systemd/system ]]; then
  echo "ERROR: systemd is required for the boot-persistent host supervisor." >&2
  echo "On WSL2, enable systemd in /etc/wsl.conf and restart WSL, then rerun this script." >&2
  exit 1
fi

SUDO=()
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || { echo "ERROR: sudo is required to install the system service" >&2; exit 1; }
  SUDO=(sudo)
fi

RUN_USER="${SUDO_USER:-${USER:-$(id -un)}}"
RUN_GROUP="$(id -gn "$RUN_USER")"
UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"
DOCKER_GROUP_LINE=""
if getent group docker >/dev/null 2>&1; then
  DOCKER_GROUP_LINE="SupplementaryGroups=docker"
  if ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx docker; then
    echo "WARNING: $RUN_USER is not currently in the docker group."
    echo "         The supervisor can report Git status, but maintenance Docker rebuilds may fail."
  fi
fi

TMP_UNIT="$(mktemp)"
trap 'rm -f "$TMP_UNIT"' EXIT
cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Ascento host maintenance supervisor
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
$DOCKER_GROUP_LINE
WorkingDirectory=$REPO_ROOT
Environment=HOME=$(getent passwd "$RUN_USER" | cut -d: -f6)
ExecStart=$PYTHON_BIN "$REPO_ROOT/scripts/host_supervisor.py" --repo "$REPO_ROOT" --socket "$SOCKET_PATH"
Restart=on-failure
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true
UMask=0007

[Install]
WantedBy=multi-user.target
EOF

"${SUDO[@]}" install -m 0644 "$TMP_UNIT" "$UNIT_PATH"
"${SUDO[@]}" systemctl daemon-reload
"${SUDO[@]}" systemctl enable --now "$SERVICE_NAME"

for _ in {1..30}; do
  [[ -S "$SOCKET_PATH" ]] && break
  sleep 0.2
done
if [[ ! -S "$SOCKET_PATH" ]]; then
  echo "ERROR: supervisor socket did not appear: $SOCKET_PATH" >&2
  "${SUDO[@]}" systemctl status "$SERVICE_NAME" --no-pager || true
  exit 1
fi

printf 'installed_at=%s\nservice=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SERVICE_NAME" \
  >"$REPO_ROOT/.maintenance/supervisor-installed"
chmod 600 "$REPO_ROOT/.maintenance/supervisor-installed"

echo "Host supervisor installed and running."
echo "Service: $SERVICE_NAME"
echo "Socket:  $SOCKET_PATH"
echo "The Dashboard receives only this restricted socket, never /var/run/docker.sock."

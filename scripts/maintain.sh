#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="https://github.com/MarcusFunt/Ascento_MuJoCo_Playground.git"
BRANCH="${ASCENTO_BRANCH:-main}"
SCRIPT_PATH="${BASH_SOURCE[0]:-}"
DEFAULT_INSTALL_DIR="$HOME/Ascento_MuJoCo_Playground"
if [[ -n "$SCRIPT_PATH" && -f "$SCRIPT_PATH" ]]; then
  SCRIPT_REPO="$(cd "$(dirname "$SCRIPT_PATH")/.." 2>/dev/null && pwd || true)"
  [[ -d "${SCRIPT_REPO:-}/.git" ]] && DEFAULT_INSTALL_DIR="$SCRIPT_REPO"
fi
INSTALL_DIR="${ASCENTO_INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"
COMPUTE="${ASCENTO_COMPUTE:-auto}"
INSTALL_SYSTEM=1
BUILD_DOCKER=1
START_DASHBOARD=1
FORCE=0

usage() {
  cat <<'EOF'
Usage: maintain.sh [options]

Bootstrap or update Ascento_MuJoCo_Playground to an exact repository state.
Existing logs/checkpoints/captures are preserved.

Options:
  --install-dir PATH       checkout location (default: current checkout or ~/Ascento_MuJoCo_Playground)
  --branch NAME            branch to install/update (default: main)
  --compute auto|cu128|cpu compute backend (default: auto)
  --skip-system-install    do not install missing OS/Docker/NVIDIA tooling
  --skip-docker-build      sync source dependencies but do not rebuild containers
  --no-start-dashboard     build containers but do not start the dashboard service
  --force                  discard dirty tracked files/local-only commits
  -h, --help               show this help
EOF
}

while (($#)); do
  case "$1" in
    --install-dir) [[ $# -ge 2 ]] || { echo "--install-dir needs a value" >&2; exit 2; }; INSTALL_DIR="$2"; shift 2 ;;
    --branch) [[ $# -ge 2 ]] || { echo "--branch needs a value" >&2; exit 2; }; BRANCH="$2"; shift 2 ;;
    --compute) [[ $# -ge 2 ]] || { echo "--compute needs a value" >&2; exit 2; }; COMPUTE="$2"; shift 2 ;;
    --skip-system-install) INSTALL_SYSTEM=0; shift ;;
    --skip-docker-build) BUILD_DOCKER=0; shift ;;
    --no-start-dashboard) START_DASHBOARD=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$COMPUTE" in auto|cu128|cpu) ;; *) echo "Invalid --compute: $COMPUTE" >&2; exit 2 ;; esac

log() { printf '\n==> %s\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

[[ "$(uname -s)" == "Linux" ]] || die "The maintained runtime is Linux/WSL2. Run this script inside Linux or WSL2."

SUDO=()
if [[ ${EUID:-$(id -u)} -ne 0 ]] && have sudo; then SUDO=(sudo); fi

apt_available() { have apt-get && [[ -r /etc/os-release ]]; }
apt_install() {
  (( INSTALL_SYSTEM )) || die "Missing system dependency and --skip-system-install was requested."
  apt_available || die "Automatic system installation currently supports Debian/Ubuntu-derived systems."
  ((${#SUDO[@]})) || [[ ${EUID:-$(id -u)} -eq 0 ]] || die "sudo is required for system installation."
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y --no-install-recommends "$@"
}

ensure_base_tools() {
  local missing=()
  have curl || missing+=(curl)
  have git || missing+=(git)
  if ((${#missing[@]})); then
    log "Installing base tools: ${missing[*]}"
    apt_install ca-certificates "${missing[@]}"
  fi
}

install_docker_engine() {
  (( INSTALL_SYSTEM )) || die "Docker/Compose is missing and system installation is disabled."
  apt_available || die "Automatic Docker installation currently supports Debian/Ubuntu-derived systems."
  # shellcheck disable=SC1091
  . /etc/os-release
  local distro="${ID:-}"
  case "$distro" in ubuntu|debian) ;; *) die "Automatic Docker installation supports Ubuntu/Debian; found ${distro:-unknown}." ;; esac
  local suite="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
  [[ -n "$suite" ]] || die "Could not determine distro codename for Docker repository."

  log "Installing Docker Engine and Compose plugin"
  apt_install ca-certificates curl
  "${SUDO[@]}" install -m 0755 -d /etc/apt/keyrings
  "${SUDO[@]}" curl -fsSL "https://download.docker.com/linux/$distro/gpg" -o /etc/apt/keyrings/docker.asc
  "${SUDO[@]}" chmod a+r /etc/apt/keyrings/docker.asc
  local arch
  arch="$(dpkg --print-architecture)"
  printf '%s\n' \
    'Types: deb' \
    "URIs: https://download.docker.com/linux/$distro" \
    "Suites: $suite" \
    'Components: stable' \
    "Architectures: $arch" \
    'Signed-By: /etc/apt/keyrings/docker.asc' \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/docker.sources >/dev/null
  "${SUDO[@]}" apt-get update
  if ! "${SUDO[@]}" apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin; then
    die "Docker package installation failed. Remove conflicting distro Docker packages, then rerun. Existing /var/lib/docker data should not be deleted."
  fi
  if have systemctl; then "${SUDO[@]}" systemctl enable --now docker || true; fi
  if [[ -n "${USER:-}" ]] && have usermod; then "${SUDO[@]}" usermod -aG docker "$USER" || true; fi
}

docker_exec() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif ((${#SUDO[@]})); then
    "${SUDO[@]}" docker "$@"
  else
    docker "$@"
  fi
}

ensure_docker() {
  if ! have docker || ! docker compose version >/dev/null 2>&1; then install_docker_engine; fi
  docker_exec info >/dev/null 2>&1 || die "Docker daemon is not reachable."
  docker_exec compose version >/dev/null 2>&1 || die "Docker Compose plugin is unavailable."
}

ensure_uv() {
  export PATH="$HOME/.local/bin:$PATH"
  if ! have uv; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    uv self update >/dev/null 2>&1 || true
  fi
  have uv || die "uv installation failed."
  uv python install 3.12
}

ensure_nvidia_container_toolkit() {
  [[ "$COMPUTE" == "cu128" ]] || return 0
  have nvidia-smi || die "CUDA backend selected but nvidia-smi is unavailable."
  if grep -qi microsoft /proc/version 2>/dev/null && docker_exec info >/dev/null 2>&1; then
    log "WSL2 detected; using the Docker/WSL GPU integration already present on the host"
    return 0
  fi
  if have nvidia-ctk && docker_exec info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; then return 0; fi
  (( INSTALL_SYSTEM )) || die "NVIDIA Container Toolkit is required for CUDA Docker use."
  log "Installing/configuring NVIDIA Container Toolkit"
  apt_install ca-certificates curl gnupg2
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | "${SUDO[@]}" gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | "${SUDO[@]}" tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y nvidia-container-toolkit
  "${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker
  if have systemctl; then "${SUDO[@]}" systemctl restart docker; else "${SUDO[@]}" service docker restart; fi
}

choose_compute() {
  if [[ "$COMPUTE" == "auto" ]]; then
    if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then COMPUTE=cu128; else COMPUTE=cpu; fi
  fi
  log "Selected compute backend: $COMPUTE"
}

OLD_COMMIT=""
OLD_BRANCH=""
prepare_checkout() {
  INSTALL_DIR="$(realpath -m "$INSTALL_DIR")"
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Updating existing checkout: $INSTALL_DIR"
    OLD_COMMIT="$(git -C "$INSTALL_DIR" rev-parse HEAD)"
    OLD_BRANCH="$(git -C "$INSTALL_DIR" branch --show-current || true)"
    local dirty
    dirty="$(git -C "$INSTALL_DIR" status --porcelain --untracked-files=no)"
    if [[ -n "$dirty" && "$FORCE" -ne 1 ]]; then
      die "Tracked files have local modifications. Commit/stash them or rerun with --force. Runs are never removed."
    fi
    git -C "$INSTALL_DIR" fetch --prune origin
    local local_only=0
    if git -C "$INSTALL_DIR" rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
      local_only="$(git -C "$INSTALL_DIR" rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo 0)"
    fi
    if [[ "$local_only" != "0" && "$FORCE" -ne 1 ]]; then
      die "The checkout has local-only commits. Merge/push them first or rerun with --force."
    fi
    git -C "$INSTALL_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
  else
    if [[ -e "$INSTALL_DIR" && -n "$(ls -A "$INSTALL_DIR" 2>/dev/null || true)" ]]; then
      die "Install directory exists but is not a Git checkout: $INSTALL_DIR"
    fi
    log "Cloning fresh checkout into $INSTALL_DIR"
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$INSTALL_DIR"
  fi
  git -C "$INSTALL_DIR" submodule sync --recursive
  git -C "$INSTALL_DIR" submodule update --init --recursive --force
}

sync_python_dependencies() {
  log "Synchronizing exact Python environment"
  local py
  py="$(uv python find 3.12)"
  local args=(sync --frozen --python "$py" --all-groups)
  while IFS= read -r extra; do
    [[ -n "$extra" ]] || continue
    [[ "$extra" == "cpu" || "$extra" == "cu128" ]] && continue
    args+=(--extra "$extra")
  done < <("$py" - "$INSTALL_DIR/pyproject.toml" <<'PY'
import sys, tomllib
with open(sys.argv[1], 'rb') as f:
    data = tomllib.load(f)
for name in (data.get('project', {}).get('optional-dependencies', {}) or {}):
    print(name)
PY
)
  args+=(--extra "$COMPUTE")
  (cd "$INSTALL_DIR" && uv "${args[@]}")
}

stamp_legacy_runs() {
  [[ -n "$OLD_COMMIT" ]] || return 0
  local root="$INSTALL_DIR/logs/rsl_rl"
  [[ -d "$root" ]] || return 0
  log "Backfilling repository provenance for legacy runs without commit metadata"
  local args=(--root "$root" --commit "$OLD_COMMIT")
  [[ -n "$OLD_BRANCH" ]] && args+=(--branch "$OLD_BRANCH")
  "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/scripts/stamp_run_provenance.py" "${args[@]}"
}

build_frontend() {
  log "Reconciling frontend dependencies and building dashboard"
  local frontend="$INSTALL_DIR/dashboard/frontend"
  local node_major=0
  if have node && have npm; then node_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"; fi
  if [[ "$node_major" =~ ^[0-9]+$ ]] && (( node_major >= 20 )); then
    (cd "$frontend" && npm ci && npm run build)
  else
    ensure_docker
    docker_exec run --rm --user "$(id -u):$(id -g)" \
      -v "$frontend:/work" -w /work node:20-bookworm-slim \
      sh -lc 'npm ci && npm run build'
  fi
}

write_maintenance_state() {
  local state="$INSTALL_DIR/.maintenance"
  mkdir -p "$state"
  local commit branch
  commit="$(git -C "$INSTALL_DIR" rev-parse HEAD)"
  branch="$(git -C "$INSTALL_DIR" branch --show-current)"
  cat >"$state/repository-version.json" <<EOF
{
  "commit": "$commit",
  "branch": "$branch",
  "compute": "$COMPUTE",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
  cat >"$state/compose.env" <<EOF
ASCENTO_COMPUTE_EXTRA=$COMPUTE
ASCENTO_REPOSITORY_COMMIT=$commit
ASCENTO_REPOSITORY_BRANCH=$branch
ASCENTO_DASHBOARD_PORT=${ASCENTO_DASHBOARD_PORT:-8000}
EOF
}

build_containers() {
  (( BUILD_DOCKER )) || return 0
  log "Rebuilding Docker image from the current lockfiles"
  local envfile="$INSTALL_DIR/.maintenance/compose.env"
  local files=(-f "$INSTALL_DIR/docker/compose.yaml")
  [[ "$COMPUTE" == "cu128" ]] && files+=(-f "$INSTALL_DIR/docker/compose.gpu.yaml")
  docker_exec compose --env-file "$envfile" "${files[@]}" down --remove-orphans || true
  docker_exec compose --env-file "$envfile" "${files[@]}" build --pull --no-cache
  if (( START_DASHBOARD )); then docker_exec compose --env-file "$envfile" "${files[@]}" up -d dashboard; fi
}

verify_installation() {
  log "Running maintenance verification"
  (cd "$INSTALL_DIR" && .venv/bin/python -m pytest -q tests_dashboard)
  if (( BUILD_DOCKER )); then
    local files=(-f "$INSTALL_DIR/docker/compose.yaml")
    [[ "$COMPUTE" == "cu128" ]] && files+=(-f "$INSTALL_DIR/docker/compose.gpu.yaml")
    docker_exec compose --env-file "$INSTALL_DIR/.maintenance/compose.env" "${files[@]}" config >/dev/null
  fi
}

ensure_base_tools
choose_compute
if (( BUILD_DOCKER )); then ensure_docker; fi
ensure_uv
if (( BUILD_DOCKER )); then ensure_nvidia_container_toolkit; fi
prepare_checkout
sync_python_dependencies
stamp_legacy_runs
build_frontend
write_maintenance_state
build_containers
verify_installation

log "Maintenance complete"
echo "Checkout: $INSTALL_DIR"
echo "Repository: $(git -C "$INSTALL_DIR" rev-parse --short HEAD) ($BRANCH)"
echo "Compute backend: $COMPUTE"
if (( BUILD_DOCKER && START_DASHBOARD )); then echo "Dashboard: http://127.0.0.1:${ASCENTO_DASHBOARD_PORT:-8000}"; fi

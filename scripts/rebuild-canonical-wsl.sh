#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/Ascento_MuJoCo_Playground"
STATE="$ROOT/.maintenance"
ENV_FILE="$STATE/compose.env"
VERSION_FILE="$STATE/repository-version.json"
REQUESTED_COMPUTE="$(printenv ASCENTO_COMPUTE 2>/dev/null || printf auto)"
SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == Linux ]] || die "Run inside WSL, not Windows."
grep -qi microsoft /proc/version || die "This script is restricted to WSL."
[[ "$SCRIPT_ROOT" == "$ROOT" ]] || die "Wrong checkout: $SCRIPT_ROOT"
[[ "$(git -C "$ROOT" rev-parse --show-toplevel)" == "$ROOT" ]] || die "Canonical Git root mismatch."
[[ "$(git -C "$ROOT" branch --show-current)" == main ]] || die "Canonical WSL checkout must be on main."
git -C "$ROOT" remote get-url origin | grep -qi Ascento_MuJoCo_Playground || die "Unexpected origin remote."
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "Checkout is dirty; commit or move changes first."
command -v flock >/dev/null || die "flock is required."
mkdir -p "$STATE"
exec 9>"$STATE/rebuild-canonical-wsl.lock"
flock -n 9 || die "Another rebuild is running."
command -v docker >/dev/null && docker info >/dev/null 2>&1 || die "Docker Desktop's WSL engine is unavailable."
docker compose version >/dev/null 2>&1 || die "Docker Compose is unavailable."

case "$REQUESTED_COMPUTE" in
  auto) if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then COMPUTE=cu128; else COMPUTE=cpu; fi ;;
  cpu|cu128) COMPUTE="$REQUESTED_COMPUTE" ;;
  *) die "ASCENTO_COMPUTE must be auto, cpu, or cu128." ;;
esac
if [[ "$COMPUTE" == cu128 ]]; then
  command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 || die "CUDA requested but WSL cannot see the GPU."
fi

log "Updating canonical WSL main with a safe fast-forward"
git -C "$ROOT" fetch --prune origin refs/heads/main:refs/remotes/origin/main
git -C "$ROOT" merge --ff-only origin/main
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "Checkout became dirty."
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$COMMIT" == "$(git -C "$ROOT" rev-parse origin/main)" ]] || die "main is not origin/main."

read_env() { awk -F= -v k="$1" '$1==k {v=substr($0,index($0,"=")+1)} END{print v}' "$ENV_FILE" 2>/dev/null || true; }
PORT="$(read_env ASCENTO_DASHBOARD_PORT)"
BIND="$(read_env ASCENTO_DASHBOARD_BIND_ADDRESS)"
[[ -n "$PORT" ]] || PORT=8000
[[ -n "$BIND" ]] || BIND=127.0.0.1
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] || die "Invalid dashboard port."
PORT_NUMBER=$((10#$PORT))
(( PORT_NUMBER > 0 && PORT_NUMBER < 65536 )) || die "Dashboard port out of range."
[[ "$BIND" =~ ^[[:alnum:].:_-]+$ ]] || die "Invalid dashboard bind address."

umask 077
TEMP_ENV="$(mktemp "$STATE/compose.env.XXXXXX")"
trap 'rm -f -- "$TEMP_ENV"' EXIT
if [[ -f "$ENV_FILE" ]]; then
  grep -Ev '^(ASCENTO_COMPUTE_EXTRA|ASCENTO_CANONICAL_SOURCE_ROOT|ASCENTO_RUNTIME_KIND|ASCENTO_REPOSITORY_COMMIT|ASCENTO_REPOSITORY_BRANCH|ASCENTO_REPOSITORY_DIRTY|ASCENTO_DASHBOARD_PORT|ASCENTO_DASHBOARD_BIND_ADDRESS)=' "$ENV_FILE" > "$TEMP_ENV" || true
fi
cat >>"$TEMP_ENV" <<EOF
ASCENTO_COMPUTE_EXTRA=$COMPUTE
ASCENTO_CANONICAL_SOURCE_ROOT=$ROOT
ASCENTO_RUNTIME_KIND=packaged-docker
ASCENTO_REPOSITORY_COMMIT=$COMMIT
ASCENTO_REPOSITORY_BRANCH=main
ASCENTO_REPOSITORY_DIRTY=0
ASCENTO_DASHBOARD_PORT=$PORT
ASCENTO_DASHBOARD_BIND_ADDRESS=$BIND
EOF
chmod 600 "$TEMP_ENV"
mv -f -- "$TEMP_ENV" "$ENV_FILE"
trap - EXIT
export ASCENTO_COMPUTE_EXTRA="$COMPUTE" ASCENTO_CANONICAL_SOURCE_ROOT="$ROOT"
export ASCENTO_RUNTIME_KIND=packaged-docker ASCENTO_REPOSITORY_COMMIT="$COMMIT"
export ASCENTO_REPOSITORY_BRANCH=main ASCENTO_REPOSITORY_DIRTY=0
export ASCENTO_DASHBOARD_PORT="$PORT" ASCENTO_DASHBOARD_BIND_ADDRESS="$BIND"

COMPOSE_FILES=(-f "$ROOT/docker/compose.yaml")
if [[ "$COMPUTE" == cu128 ]]; then COMPOSE_FILES+=(-f "$ROOT/docker/compose.gpu.yaml"); fi
touch "$STATE/tailscale-enabled"
COMPOSE_FILES+=(-f "$ROOT/docker/compose.tailscale.yaml")
compose() {
  local env_file="$1"
  shift
  docker compose --project-name docker --env-file "$env_file" "${COMPOSE_FILES[@]}" "$@"
}

log "Validating Compose configuration: $COMMIT ($COMPUTE)"
compose "$ENV_FILE" config --quiet
log "Building dashboard and CLI image from WSL source"
compose "$ENV_FILE" build --pull dashboard ascento-mjlab
log "Starting persistent database and Tailscale services"
compose "$ENV_FILE" up -d dashboard-db tailscale-dashboard
log "Recreating dashboard from the new image"
compose "$ENV_FILE" up -d --no-deps --force-recreate --wait --wait-timeout 180 dashboard

cat >"$VERSION_FILE" <<EOF
{
  "commit": "$COMMIT",
  "branch": "main",
  "compute": "$COMPUTE",
  "runtime_environment": "/opt/venv",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
chmod 600 "$VERSION_FILE"

log "Verifying live API provenance and built frontend"
CHECK='import json,sys,time,urllib.error as e,urllib.request as u
c,g=sys.argv[1:3]
base="http://127.0.0.1:8000"
def get(path):
    for _ in range(60):
        try:
            return u.urlopen(base+path,timeout=3).read()
        except (OSError,e.URLError):
            time.sleep(1)
    raise SystemExit("Dashboard API did not become reachable within 60 seconds")
h=json.loads(get("/api/health"))
v=json.loads(get("/api/runtime/identity"))
assert h["status"]=="healthy",h
assert v["checkout"]["commit"]==c and v["checkout"]["origin_main"]==c,v
assert v["packaged_image"]["commit"]==c and v["packaged_image"]["manifest_commit"]==c,v
assert v["packaged_image"]["compute"]==g,v
assert v["comparisons"]["checkout_matches_image"] and v["comparisons"]["image_matches_manifest"],v
html=get("/").decode()
a=html.split("src="+chr(34),1)[1].split(chr(34),1)[0]
assert a.startswith("/assets/"),html
assert len(get(a))>100
print(json.dumps({"health":h["status"],"commit":c,"compute":g,"frontend_asset":a,"runtime_matches":True},indent=2))'
compose "$ENV_FILE" exec -T dashboard python -c "$CHECK" "$COMMIT" "$COMPUTE"
TS_CONTAINER="$(compose "$ENV_FILE" ps -q tailscale-dashboard)"
[[ -n "$TS_CONTAINER" ]] || die "Tailscale sidecar is missing."
TS_IP="$(docker exec "$TS_CONTAINER" tailscale ip -4)"
[[ "$TS_IP" =~ ^100\. ]] || die "Tailscale has no tailnet IPv4 address."
IMAGE_ID="$(docker image inspect ascento-mjlab:local --format '{{.Id}}')"
log "Rebuild and restart complete"
printf 'Canonical checkout: %s\nCommit: %s\nCompute: %s\nImage: %s\nDashboard: http://%s:8000\n' "$ROOT" "$COMMIT" "$COMPUTE" "$IMAGE_ID" "$TS_IP"

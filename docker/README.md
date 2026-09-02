# Docker workflow

The development image contains Python, JAX/MJX, Brax, MuJoCo, the dashboard,
and the headless EGL runtime. Compose mounts the checkout at `/workspace`, so
Python source changes do not require an image rebuild. Training artifacts are
stored under `docker-data/artifacts` on the host and mounted at `/artifacts`.

## Build and use the development container

```bash
docker compose build dev
docker compose run --rm dev pytest -q tests
```

Run a monitored training job with the persistent artifact root:

```bash
docker compose run --rm trainer \
  python -m dashboard.launch \
  --artifact-root /artifacts/runs \
  -- \
  --stage balance \
  --timesteps 50000000 \
  --num-envs 1024
```

Run the complete local acceptance sequence with `./scripts/docker-smoke.sh`.
It requires a working NVIDIA Container Toolkit setup and runs CPU tests plus a
small GPU training/rendering check.

## Dashboard

```bash
docker compose up dashboard
```

Open `http://127.0.0.1:8000`. To expose it privately to the tailnet from the
host, use `tailscale serve --bg 8000`; do not use Tailscale Funnel.

## Snapshot image

Build the dependency image first, then build a source snapshot tagged with the
full Git SHA:

```bash
GIT_SHA="$(git rev-parse HEAD)"
docker build \
  --build-arg GIT_COMMIT="$GIT_SHA" \
  --build-arg DOCKER_IMAGE="ascento:$GIT_SHA" \
  -f docker/Dockerfile.snapshot \
  -t "ascento:$GIT_SHA" \
  .
```

Snapshot runs should mount only `./docker-data/artifacts:/artifacts`; do not
bind-mount the checkout over `/workspace`.

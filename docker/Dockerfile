FROM node:20-bookworm-slim AS dashboard-frontend
WORKDIR /frontend
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json ./
RUN npm ci
COPY dashboard/frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm

ARG UV_VERSION=0.12.6
ARG COMPUTE_EXTRA=cu128
ARG REPOSITORY_COMMIT=unknown
ARG REPOSITORY_BRANCH=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=disable \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    ASCENTO_REPOSITORY_COMMIT=$REPOSITORY_COMMIT \
    ASCENTO_REPOSITORY_BRANCH=$REPOSITORY_BRANCH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglfw3 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY tests_mjlab ./tests_mjlab
COPY tests_dashboard ./tests_dashboard
COPY scripts ./scripts
COPY --from=dashboard-frontend /frontend/dist ./dashboard/frontend/dist

# Install every optional extra that is compatible with the selected compute
# backend, plus every dependency group. cpu/cu128 are mutually exclusive.
RUN set -eu; \
    EXTRAS="$(python - "$COMPUTE_EXTRA" <<'PY'
import sys, tomllib
with open('pyproject.toml', 'rb') as handle:
    data = tomllib.load(handle)
compute = sys.argv[1]
extras = []
for name in (data.get('project', {}).get('optional-dependencies', {}) or {}):
    if name in {'cpu', 'cu128'}:
        continue
    extras.extend(['--extra', name])
extras.extend(['--extra', compute])
print(' '.join(extras))
PY
)"; \
    uv sync --frozen --all-groups $EXTRAS

COPY docker/smoke.sh /usr/local/bin/ascento-smoke
RUN chmod +x /usr/local/bin/ascento-smoke \
    && mkdir -p /workspace/logs /workspace/checkpoints /workspace/captures

CMD ["ascento-smoke"]

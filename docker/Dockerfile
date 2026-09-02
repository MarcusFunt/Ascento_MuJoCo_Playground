# The development image contains the complete simulation, training, dashboard,
# and headless rendering environment. Source code is mounted by Compose.

FROM node:20-bookworm-slim AS dashboard-frontend

WORKDIR /src/dashboard/frontend

COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json ./
RUN npm ci

COPY dashboard/frontend/ ./
RUN npm run build


FROM python:3.12-slim-bookworm AS ascento-dev-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ASCENTO_JAX_PLATFORM=cuda \
    MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libegl1 \
    libgles2 \
    libgl1 \
    libglvnd0 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-cuda.txt /tmp/requirements-cuda.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt
COPY requirements-dashboard.txt /tmp/requirements-dashboard.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      -r /tmp/requirements-cuda.txt \
      -r /tmp/requirements-dev.txt \
      -r /tmp/requirements-dashboard.txt

RUN mkdir -p /artifacts /opt/ascento-dashboard-dist


FROM ascento-dev-base AS ascento-dev

COPY --from=dashboard-frontend /src/dashboard/frontend/dist /opt/ascento-dashboard-dist
ENV ASCENTO_DASHBOARD_DIST=/opt/ascento-dashboard-dist

CMD ["bash"]

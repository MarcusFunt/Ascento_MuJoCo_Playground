FROM python:3.12-slim-bookworm

ARG UV_VERSION=0.12.6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests_mjlab ./tests_mjlab
RUN uv sync --frozen --extra cu128 --extra dashboard --no-dev

COPY docker/smoke.sh /usr/local/bin/ascento-smoke
RUN chmod +x /usr/local/bin/ascento-smoke && mkdir -p /workspace/logs /workspace/checkpoints /workspace/captures

ENTRYPOINT ["ascento-smoke"]

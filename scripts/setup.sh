#!/usr/bin/env bash
# Create the CUDA-enabled Python environment used by this repository.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="python3"
INSTALL_DEV=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--venv PATH] [--python PATH] [--dev]

Creates a virtual environment and installs the pinned CUDA 12/JAX/MJX/Brax
dependencies. The default virtual environment is .venv in the repository.

Options:
  --venv PATH    Virtual-environment directory (default: .venv)
  --python PATH  Python 3.10+ interpreter to use (default: python3)
  --dev          Also install developer test tooling
  -h, --help     Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dev)
      INSTALL_DEV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${PYTHON_VERSION}" in
  3.10|3.11|3.12|3.13) ;;
  *)
    echo "Python 3.10 through 3.13 is required; found ${PYTHON_VERSION}." >&2
    exit 1
    ;;
esac

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements-cuda.txt"

if [[ "${INSTALL_DEV}" -eq 1 ]]; then
  "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements-dev.txt"
fi

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VENV_PYTHON}" - <<'PY'
import brax
import jax
import mujoco
import numpy
from mujoco import mjx

print("Environment ready")
print(f"  Python: {__import__('sys').version.split()[0]}")
print(f"  JAX: {jax.__version__} ({jax.default_backend()})")
print(f"  MuJoCo: {mujoco.__version__}")
print(f"  Brax: {brax.__version__}")
print(f"  NumPy: {numpy.__version__}")
print(f"  Devices: {jax.devices()}")
print(f"  MJX available: {mjx is not None}")
PY

cat <<EOF

Activate it with:
  source "${VENV_DIR}/bin/activate"

For CUDA training:
  export ASCENTO_JAX_PLATFORM=cuda
  python -m training.train
EOF

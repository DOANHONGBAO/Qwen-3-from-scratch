#!/usr/bin/env bash
# Create (or update) the local CPU-only conda environment and validate it.
#
#   bash scripts/setup_env.sh
#
# Requires: conda (Miniforge/Miniconda) on PATH. No sudo, no system packages.
# Never downloads model weights.
set -euo pipefail

ENV_NAME="qwen3-scratch"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniforge first: https://github.com/conda-forge/miniforge" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo ">> Updating existing environment '${ENV_NAME}'"
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  echo ">> Creating environment '${ENV_NAME}'"
  conda env create -f environment.yml
fi

echo ">> Validating"
conda run -n "${ENV_NAME}" python scripts/check_env.py
conda run -n "${ENV_NAME}" ruff check .
conda run -n "${ENV_NAME}" ruff format --check .
conda run -n "${ENV_NAME}" pytest
echo ">> Done. Activate with: conda activate ${ENV_NAME}"

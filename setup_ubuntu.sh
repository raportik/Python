#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected, installing CUDA torch..."
  # Default to CUDA 12.1 wheels
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  echo "No GPU detected, installing CPU torch..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

echo "Done. Activate venv with: source .venv/bin/activate"

#!/usr/bin/env bash
# Create the fine-tuning environment on a GPU node. Run once.
#   bash hpc/setup_env.sh [cu124|cu126|cu128]
set -euo pipefail

CUDA_TAG="${1:-cu126}"
cd "$(dirname "$0")/.."

uv venv --python 3.12 .venv
# CUDA torch first, from the PyTorch index; the train extra then reuses it.
uv pip install --python .venv/bin/python torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
uv pip install --python .venv/bin/python -e ".[train]"
# Uncomment for 4-bit QLoRA on smaller GPUs:
# uv pip install --python .venv/bin/python -e ".[qlora]"

.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

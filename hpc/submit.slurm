#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1,tmpfs:20G
#SBATCH --time=16:00:00
#SBATCH --job-name=structllm-sft

# LoRA fine-tune of Qwen3-8B on the exported StructSub conversations. Nothing
# here talks to a StructSub server: the tool calls and results are already in
# train.jsonl / eval.jsonl.
#
# Submit from the project root (the directory holding pyproject.toml):
#
#   sbatch hpc/submit.slurm
#
# Everything is overridable through the environment:
#
#   sbatch --export=ALL,SFT=data/sft/<export>,OUT=runs/run2,EPOCHS=5 hpc/submit.slurm
#   sbatch --export=ALL,QLORA=1 hpc/submit.slurm          # 4-bit, for a smaller GPU
#
# GPU memory: Qwen3-8B in bf16 with LoRA and gradient checkpointing. The loss is
# computed only over the target turn, so memory grows with the prompt, not with the
# 150k-token vocabulary. At the default MAX_LENGTH=32768 (99% of turns fit) request an
# 80 GB card. A 40-48 GB card should use MAX_LENGTH=24576 (94% fit) or QLORA=1.
# Expect roughly 4-9 hours for 3 epochs on one A100/H100; the estimate is not measured. Choose the card with --gres=gpu:<type>:1 if your
# site names them. Output goes to slurm-<jobid>.out in the submit directory.
#
# train.py evaluates inside the loop (eval loss and generated-turn behaviour),
# checkpoints to OUT/last and OUT/best, and resumes from OUT/last. Resubmit the same
# command after a wall-clock kill. Overrides: EVAL_EVERY, BEHAVIOUR_SAMPLES,
# MAX_NEW_TOKENS, PATIENCE. Each behaviour evaluation generates BEHAVIOUR_SAMPLES
# turns with the 8B model and takes a few minutes; lower it to speed a run up.

module purge
module load ucl-stack/2025-12
module load python/3.11.6/gcc-12.3.0-avx2

nvidia-smi
echo "tmpdir    : ${TMPDIR:-unset}"

# Slurm runs a spooled copy of this script, so $BASH_SOURCE is no use for finding
# the project; SLURM_SUBMIT_DIR is where sbatch was run.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT}" || { echo "cannot enter ${PROJECT}" >&2; exit 1; }
if [ ! -f pyproject.toml ] || [ ! -d structllm ]; then
    echo "${PROJECT} is not the structllm project root." >&2
    echo "Submit from it, or: sbatch --export=ALL,PROJECT=/path/to/structllm hpc/submit.slurm" >&2
    exit 1
fi

# Before `set -u`: activate scripts and module functions read unset variables.
VENV="${VENV:-${PROJECT}/.venv-train}"
CUDA_TAG="${CUDA_TAG:-cu126}"

# Model weights go to Scratch, not the small home quota. Override HF_HOME freely.
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/Scratch}/hf}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

# First run builds the environment; later runs reuse it. Compute nodes need
# internet for this (PyPI, the PyTorch index, Hugging Face). If yours have none,
# run this script once on a login node with SETUP_ONLY=1, then submit as usual.
if [ ! -f "${VENV}/.structllm-ready" ]; then
    echo "building environment in ${VENV}"
    python -m venv "${VENV}"
    source "${VENV}/bin/activate"
    pip install --upgrade pip
    pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    pip install -e ".[train]"
    [ "${QLORA:-0}" = "1" ] && pip install -e ".[qlora]"
    touch "${VENV}/.structllm-ready"
else
    source "${VENV}/bin/activate"
fi

set -euo pipefail

SFT="${SFT:-}"
if [ -z "${SFT}" ]; then
    # Newest export under data/sft, if the caller did not name one.
    SFT="$(ls -dt data/sft/*/ 2>/dev/null | head -n 1 || true)"
    SFT="${SFT%/}"
fi
MODEL="${MODEL:-Qwen/Qwen3-8B}"
OUT="${OUT:-runs/$(date +%Y%m%d-%H%M)-${SLURM_JOB_ID:-local}}"
EPOCHS="${EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
EXTRA="${EXTRA:-}"
[ "${QLORA:-0}" = "1" ] && EXTRA="${EXTRA} --qlora"

if [ -z "${SFT}" ] || [ ! -f "${SFT}/train.jsonl" ] || [ ! -f "${SFT}/eval.jsonl" ]; then
    echo "no SFT export found (looked for '${SFT:-data/sft/*}')." >&2
    echo "Copy one here, e.g.:" >&2
    echo "  rsync -a /Volumes/X9/scratch/structsub-mcp-dev/sft/<export> <cluster>:${PROJECT}/data/sft/" >&2
    exit 1
fi

if [ "${SETUP_ONLY:-0}" = "1" ]; then
    # Fetch the weights and tokenizer now, from a login node, then stop.
    python -c "from huggingface_hub import snapshot_download as s; print(s('${MODEL}'))"
    echo "setup done; submit without SETUP_ONLY"
    exit 0
fi

mkdir -p "${OUT}"
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "project   : ${PROJECT}"
echo "venv      : ${VENV}"
echo "model     : ${MODEL}"
echo "data      : ${SFT}"
echo "out       : ${OUT}"

# Fail now, not after the model has loaded: a run that lands on the CPU spends
# the wall clock and finishes nothing.
python - <<'PY'
import sys

import torch

print("torch     :", torch.__version__, "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("no GPU visible to torch; check the CUDA_TAG and the node")
props = torch.cuda.get_device_properties(0)
print("device    :", props.name, f"{props.total_memory / 2**30:.0f} GiB")
PY

# Tokenise and report lengths, skipped turns and totals. Cheap, and a bad export
# shows up here in seconds rather than at the first training step.
python -m structllm.train --data "${SFT}" --model "${MODEL}" --output-dir "${OUT}" \
    --max-length "${MAX_LENGTH}" --prepare-only

# Evaluation is part of the loop: eval loss plus generated-turn behaviour on a fixed
# sample, at the end of every epoch (or every EVAL_EVERY steps). The adapter is
# checkpointed to ${OUT}/last, the best one to ${OUT}/best, and every evaluation is
# appended to ${OUT}/metrics.jsonl. --resume continues from ${OUT}/last when it
# exists, so resubmitting the same command after a wall-clock kill continues the run.
python -m structllm.train \
    --data "${SFT}" \
    --model "${MODEL}" \
    --output-dir "${OUT}" \
    --epochs "${EPOCHS}" \
    --max-length "${MAX_LENGTH}" \
    --eval-every "${EVAL_EVERY:-0}" \
    --behaviour-samples "${BEHAVIOUR_SAMPLES:-32}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-768}" \
    --patience "${PATIENCE:-0}" \
    --resume \
    ${EXTRA}

if [ ! -f "${OUT}/adapter_model.safetensors" ]; then
    echo "training finished without writing an adapter to ${OUT}" >&2
    exit 1
fi
echo "adapter   : ${OUT} (final) and ${OUT}/best (highest behaviour score)"
echo "metrics   :"
cat "${OUT}/metrics.jsonl" 2>/dev/null | tail -n 20
ls -lh "${OUT}"

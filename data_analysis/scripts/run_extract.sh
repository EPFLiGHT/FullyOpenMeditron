#!/bin/bash
#
# run_extract.sh — Stage 1 (metadata extraction) SLURM wrapper
#
# Originally developed on CSCS Clariden. Adapt the #SBATCH directives for
# your scheduler. Submit from the parent repo root:
#
#   mkdir -p logs                                     # one-time, before first sbatch
#   sbatch data_analysis/scripts/run_extract.sh
#
# Before running on a different cluster, edit:
#   - --account            : your SLURM account
#   - --environment        : path to your vLLM container / activation
#                            (CSCS pyxis uses .toml; other clusters use
#                            singularity, modules, or conda activation)
#   - --partition / --time : per your cluster's queue policies
#

#SBATCH --job-name=metadata-full
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu
#SBATCH --environment=PATH_TO_YOUR_VLLM_CONTAINER
#SBATCH --output=logs/metadata_full_%j.out
#SBATCH --error=logs/metadata_full_%j.err

set -euo pipefail

# CSCS-specific: pyxis-injected SSL_CERT_FILE breaks pip on some Clariden
# nodes. Harmless to leave in elsewhere.
unset SSL_CERT_FILE

# Move to the repo root regardless of where sbatch was invoked from.
# This script lives at: <repo>/data_analysis/scripts/run_extract.sh
cd "$(dirname "$0")/../.."

mkdir -p logs

# Persist HF cache on a fast scratch volume so Qwen3-32B (~64 GB) isn't
# re-downloaded into ephemeral storage between jobs. Override HF_HOME
# externally if your cluster doesn't expose $SCRATCH.
export HF_HOME=${HF_HOME:-${SCRATCH:-$HOME/.cache/huggingface}/hf}
mkdir -p "$HF_HOME"

python3 data_analysis/scripts/01_extract_metadata.py \
  --model Qwen/Qwen3-32B \
  --tp 2 \
  --gpu-util 0.90 \
  --max-model-len 16384 \
  --max-tokens 1024 \
  --temperature 0.1 \
  --root . \
  --out-dir data_analysis/outputs \
  --n-per-group -1 \
  --seed 42 \
  --flush-every 50
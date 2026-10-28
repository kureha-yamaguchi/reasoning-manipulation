#!/usr/bin/env bash
# Run one data-parallel shard of utils.heuristic.resample_quadrants on one GPU.
# Model weights are read from a fast (non-persistent) HF cache; results go to the
# persistent results dir. Existing per-CoT output files are skipped, so re-running resumes.
#
# Usage: bash bash_scripts/resample_quadrants_shard.sh <model_name> <gpu> <shard k/K> <n> [extra args...]
#   e.g. bash bash_scripts/resample_quadrants_shard.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 0 0/3 34
set -euo pipefail

MODEL_NAME=$1; GPU=$2; SHARD=$3; N=$4; shift 4

export HF_HUB_CACHE="${FAST_HF_HUB_CACHE:-/scratch/hf/hub}"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="logs/resample_quadrants"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${MODEL_NAME//\//_}_shard${SHARD//\//of}.log"
echo "Log file: $LOG_FILE"

uv run python -m utils.heuristic.resample_quadrants \
    --model_name "$MODEL_NAME" --repetitions 10 --x_threshold 0.03 --y_threshold 0.43 \
    --n "$N" --shard "$SHARD" --max_model_len 32768 "$@" > "$LOG_FILE" 2>&1

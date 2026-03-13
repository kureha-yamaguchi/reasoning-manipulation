#!/usr/bin/env bash
set -euo pipefail

# MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# MODEL_NAME=Qwen/Qwen3-8B
BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
TRANSFER_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
TRANSFER_MODEL2=openrouter/deepseek/deepseek-r1

REPETITION=15

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/transfer_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# PROMPT_INDEX=174

# echo "=== Step 1: Perform resampling ==="
# uv run -m utils.heuristic.resample_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

# echo "=== Step 2: Compute scores ==="
# uv run -m utils.heuristic.compute_scores_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" 

# echo "=== Step 3: Plot results ==="
# uv run -m utils.heuristic.plot_sentence1_transfer_3models --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

# PROMPT_INDEX=292

# echo "=== Step 1: Perform resampling ==="
# uv run -m utils.heuristic.resample_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

# echo "=== Step 2: Compute scores ==="
# uv run -m utils.heuristic.compute_scores_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" 

# echo "=== Step 3: Plot results ==="
# uv run -m utils.heuristic.plot_sentence1_transfer_3models --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

PROMPT_INDEX=595

echo "=== Step 1: Perform resampling ==="
uv run -m utils.heuristic.resample_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

echo "=== Step 2: Compute scores ==="
uv run -m utils.heuristic.compute_scores_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" 

echo "=== Step 3: Plot results ==="
uv run -m utils.heuristic.plot_sentence1_transfer_3models --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

PROMPT_INDEX=621

echo "=== Step 1: Perform resampling ==="
uv run -m utils.heuristic.resample_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

echo "=== Step 2: Compute scores ==="
uv run -m utils.heuristic.compute_scores_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" 

echo "=== Step 3: Plot results ==="
uv run -m utils.heuristic.plot_sentence1_transfer_3models --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --transfer_model_2 "$TRANSFER_MODEL2" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"
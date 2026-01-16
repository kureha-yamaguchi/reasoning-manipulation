#!/usr/bin/env bash
set -euo pipefail

# MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# MODEL_NAME=Qwen/Qwen3-8B
# MODEL_NAME=openai/gpt-oss-20b

REPETITION=15

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/resample_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

INDEX_NUM=3

COT_NUM=1
echo "Performing resampling for CoT Number: $COT_NUM"

echo "=== Step 1: Perform resampling ==="
uv run -m utils.heuristic.resample_quadrants_better --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

echo "=== Step 2: Compute scores ==="
uv run -m utils.heuristic.compute_scores --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM"

echo "=== Step 3: Plot figure ==="
uv run -m utils.heuristic.plot_combined --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

COT_NUM=2
echo "Performing resampling for CoT Number: $COT_NUM"

echo "=== Step 1: Perform resampling ==="
uv run -m utils.heuristic.resample_quadrants_better --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

echo "=== Step 2: Compute scores ==="
uv run -m utils.heuristic.compute_scores --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM"

echo "=== Step 3: Plot figure ==="
uv run -m utils.heuristic.plot_combined --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

COT_NUM=3
echo "Performing resampling for CoT Number: $COT_NUM"

echo "=== Step 1: Perform resampling ==="
uv run -m utils.heuristic.resample_quadrants_better --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

echo "=== Step 2: Compute scores ==="
uv run -m utils.heuristic.compute_scores --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM"

echo "=== Step 3: Plot figure ==="
uv run -m utils.heuristic.plot_combined --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

COT_NUM=4
echo "Performing resampling for CoT Number: $COT_NUM"

echo "=== Step 1: Perform resampling ==="
uv run -m utils.heuristic.resample_quadrants_better --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

echo "=== Step 2: Compute scores ==="
uv run -m utils.heuristic.compute_scores --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM"

echo "=== Step 3: Plot figure ==="
uv run -m utils.heuristic.plot_combined --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

COT_NUM=5
echo "Performing resampling for CoT Number: $COT_NUM"

echo "=== Step 1: Perform resampling ==="
uv run -m utils.heuristic.resample_quadrants_better --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

echo "=== Step 2: Compute scores ==="
uv run -m utils.heuristic.compute_scores --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM"

echo "=== Step 3: Plot figure ==="
uv run -m utils.heuristic.plot_combined --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"

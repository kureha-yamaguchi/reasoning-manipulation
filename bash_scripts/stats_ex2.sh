#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME1=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
MODEL_NAME2=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_NAME3=Qwen/Qwen3-8B
MODEL_NAME4=openai/gpt-oss-20b

# INPUT_CSV=train_harmful_prompts_cot5_out5_rollout_s1_fast.csv
# INPUT_DIR=dataset

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/rollout_fast_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""


# echo "=== Step 1: Perform rollouts from s1 ==="
# uv run -m utils.heuristic.rollout_s1_fast --model_name "$MODEL_NAME1"

# echo "=== Step 1: Perform rollouts from s1 ==="
# uv run -m utils.heuristic.rollout_s1_fast --model_name "$MODEL_NAME2"

echo "=== Step 1: Perform rollouts from s1 ==="
uv run -m utils.heuristic.rollout_s1_fast --model_name "$MODEL_NAME3"

# echo "=== Step 1: Perform rollouts from s1 ==="
# uv run -m utils.heuristic.rollout_s1_fast --model_name "$MODEL_NAME4" --max_new_tokens 4096

# echo "=== Step 2: Compute scores ==="
# uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_csv "$INPUT_CSV" --input_dir "$INPUT_DIR"



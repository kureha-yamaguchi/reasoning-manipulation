#!/usr/bin/env bash
set -euo pipefail

MODEL_NAMES=(
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # Qwen/Qwen3-8B
    openai/gpt-oss-20b
)

CSV='combined_datasets_rollout_s1.csv'
HF_NAME="kureha295"

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/rollouts_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

export STRONGREJECT_VLLM_URL=http://localhost:8000/v1


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "=== Compute score outputs: $MODEL_NAME ==="
    uv run -m utils.csv_from_hf --hf_username "$HF_NAME" --dataset "$CSV" --model_name "$MODEL_NAME"
    uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_csv "$CSV" --input_dir "dataset"
    uv run -m utils.repair_scores --scored_csv "results/$MODEL_NAME/dataset/scored_combined_datasets_rollout_s1.csv"
    uv run -m utils.csv_to_hf --hf_username "$HF_NAME" --dataset "scored_$CSV" --model_name "$MODEL_NAME"
done
Collapse
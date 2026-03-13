#!/usr/bin/env bash
set -euo pipefail

MODEL_NAMES=(
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    Qwen/Qwen3-8B
    openai/gpt-oss-20b
)

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/orbench_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

export STRONGREJECT_VLLM_URL=http://localhost:8000/v1
# export OPENBLAS_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export HF_DATASETS_NUM_PROC=1
# export TOKENIZERS_PARALLELISM=false


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "=== Compute score outputs: $MODEL_NAME ==="
    uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_csv "orbench_extra_prompts_cot5_out5.csv" --input_dir dataset
    uv run -m utils.repair_scores --scored_csv results/$MODEL_NAME/dataset/scored_orbench_extra_prompts_cot5_out5.csv
    uv run -m utils.csv_to_hf --dataset scored_orbench_extra_prompts_cot5_out5.csv --model_name $MODEL_NAME
done
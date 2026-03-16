#!/usr/bin/env bash
set -euo pipefail

MODEL_NAMES=(
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    Qwen/Qwen3-8B
    openai/gpt-oss-20b
)

REPETITION=15
NUM_ROWS=30

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

export STRONGREJECT_VLLM_URL=http://localhost:8000/v1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HF_DATASETS_NUM_PROC=1
export TOKENIZERS_PARALLELISM=false

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "=== Compute score outputs: $MODEL_NAME ==="

    echo "=== Step 1: Perform resampling ==="
    uv run -m utils.heuristic.rollout_full_cot --model_name "$MODEL_NAME" --repetitions $REPETITION --n $NUM_ROWS

    echo "=== Step 2: Compute scores ==="
    uv run -m utils.heuristic.compute_scores_rollouts --model_name "$MODEL_NAME" --repetitions $REPETITION

done
# echo "=== Step 3: Plot figure ==="
# uv run -m utils.heuristic.plot_resample_graph --model_name "$MODEL_NAME" --index_number "$INDEX_NUM" --cot_number "$COT_NUM" --repetitions "$REPETITION"



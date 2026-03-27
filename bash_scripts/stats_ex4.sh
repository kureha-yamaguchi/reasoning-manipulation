#!/usr/bin/env bash
set -euo pipefail

# # Terminal 1: Start vLLM server once
# CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.5


MODEL_NAMES=(
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    Qwen/Qwen3-8B
    openai/gpt-oss-20b
)

REPETITION=10
PROMPT_INDEX=(
    174,246,501,621,806,1022,1073,1236
    1,23,65,170,181,287,339,604
    58,461,490,503,738,851,925,1299
    113,116,282,464,571,813,923,1072
)



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

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    PIDX="${PROMPT_INDEX[$i]}"
    echo "=== Compute score outputs: $MODEL_NAME (prompt_index=$PIDX) ==="
 
    echo "=== Step 1: Perform resampling ==="
    uv run -m utils.heuristic.resample_quadrants --model_name "$MODEL_NAME" --repetitions $REPETITION --prompt_index "$PIDX"
 
    echo "=== Step 2: Compute scores ==="
    uv run -m utils.heuristic.compute_scores_rollouts --model_name "$MODEL_NAME" --repetitions $REPETITION --quadrant
 
    echo "=== Step 3: Plot figure ==="
    uv run -m utils.heuristic.plot_quadrant_matrix --model_name "$MODEL_NAME" --repetitions "$REPETITION"
 
done
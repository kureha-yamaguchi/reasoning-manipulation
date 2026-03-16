#!/usr/bin/env bash
set -euo pipefail

# MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# MODEL_NAME=Qwen/Qwen3-8B
BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
TRANSFER_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

REPETITION=15

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/transfer_${TIMESTAMP}.log"

export STRONGREJECT_VLLM_URL=http://localhost:8000/v1
export HF_DATASETS_NUM_PROC=4
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

PROMPT_INDICES=(174 246 309 501 621 806 1022 1073 1236)

for PROMPT_INDEX in "${PROMPT_INDICES[@]}"; do
    echo "========================================"
    echo "Processing prompt index: $PROMPT_INDEX"
    echo "========================================"

    echo "=== Step 1: Perform resampling ==="
    uv run -m utils.heuristic.resample_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

    echo "=== Step 2: Compute scores ==="
    uv run -m utils.heuristic.compute_scores_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --prompt_index "$PROMPT_INDEX" 

    echo "=== Step 3: Plot results ==="
    uv run -m utils.heuristic.plot_sentence1_transfer --base_model "$BASE_MODEL" --transfer_model "$TRANSFER_MODEL" --prompt_index "$PROMPT_INDEX" --repetitions "$REPETITION"

    echo ""
done

echo "Finished at: $(date)"

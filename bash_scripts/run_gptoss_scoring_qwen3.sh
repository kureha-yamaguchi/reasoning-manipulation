#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# E2E pipeline for Qwen3-8B using GPT-OSS-20B (local vLLM) as the strongREJECT
# scorer, to compare against the original API-based scoring results.
#

# This script runs the full pipeline:
#   filter -> activations -> ortho -> generate -> score -> plot
# and manages the vLLM server lifecycle (kill for GPU-heavy steps, restart for scoring)
# =============================================================================

MODEL_NAME="Qwen/Qwen3-8B"
TYPE1="cot"
TYPE2="baseline"
TRY_LAYERS="13,15,17,19,21,23"

# Filtering thresholds (defaults: REFUSAL_THRESHOLD=0.05, NONREFUSAL_THRESHOLD=0.6)
REFUSAL_THRESHOLD="${REFUSAL_THRESHOLD:-0.2}"
NONREFUSAL_THRESHOLD="${NONREFUSAL_THRESHOLD:-0.8}"

RESULTS_BASE="results"
MODEL_DIR="${RESULTS_BASE}/${MODEL_NAME}"
BACKUP_DIR="${RESULTS_BASE}/Qwen/Qwen3-8B_api_scored"

VLLM_MODEL="openai/gpt-oss-20b"
VLLM_PORT=8000
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_TP_SIZE=4
REPAIR_RETRIES=30

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/gptoss_qwen3_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Model: $MODEL_NAME | Type1: $TYPE1 | Type2: $TYPE2 | Layers: $TRY_LAYERS"
echo "Thresholds: refusal=${REFUSAL_THRESHOLD}, nonrefusal=${NONREFUSAL_THRESHOLD}"
echo "Scoring method: GPT-OSS-20B via local vLLM"
echo ""

# ---- Helper functions ----

free_gpus() {
    echo "Freeing GPUs..."

    # Kill vLLM server
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    # Kill any orphaned python processes still holding GPU memory
    # (e.g. strongreject multiprocessing workers that outlive their parent)
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$gpu_pids" ]; then
        echo "  Found stale GPU processes: $gpu_pids"
        for pid in $gpu_pids; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 2
    fi

    # Verify GPUs are clear
    local remaining
    remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [ -n "$remaining" ]; then
        echo "  WARNING: Some GPU processes still alive: $remaining"
    else
        echo "  GPUs clear."
    fi
}

start_vllm_server() {
    echo "Starting vLLM server for ${VLLM_MODEL}..."
    VLLM_LOG="${LOG_DIR}/vllm_${TIMESTAMP}.log"
    python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --max-model-len 8192 > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    # Wait for server to become healthy
    echo "Waiting for vLLM server to be ready (log: $VLLM_LOG)..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server ready (took ~${i}s)"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM server failed to start within 600s. Check $VLLM_LOG"
    exit 1
}

ensure_vllm_server() {
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server already running."
    else
        start_vllm_server
    fi
    export STRONGREJECT_VLLM_URL="$VLLM_URL"
}

# ---- Preflight check ----

if [ ! -f "${MODEL_DIR}/dataset/scored_train_harmful_prompts_cot5_out5.csv" ]; then
    echo "ERROR: ${MODEL_DIR}/dataset/scored_train_harmful_prompts_cot5_out5.csv not found."
    echo "Complete the manual setup steps in the header before running."
    exit 1
fi

if [ ! -f "${MODEL_DIR}/dataset/train_harmful_prompts_cot5_out5.csv" ]; then
    echo "ERROR: Raw generation CSVs not found in ${MODEL_DIR}/dataset/."
    echo "Make sure you symlinked them from the backup dir."
    exit 1
fi

echo ""

# =============================================================================
# PHASE 1: Score test set & filter datasets (needs vLLM server, no heavy GPU)
# =============================================================================
# echo "====== PHASE 1: SCORE TEST SET & FILTER DATASETS ======"

# ensure_vllm_server

# echo "=== Step 2: Score test outputs with GPT-OSS ==="
# uv run -m utils.compute_score_outputs \
#     --model_name "$MODEL_NAME" \
#     --input_csv "test_harmful_prompts_cot5_out5.csv" \
#     --input_dir dataset

# echo "=== Step 2b: Repair NaN scores (test) ==="
# uv run -m utils.repair_scores \
#     --scored_csv "${MODEL_DIR}/dataset/scored_test_harmful_prompts_cot5_out5.csv" \
#     --max_retries "$REPAIR_RETRIES"

echo "=== Step 3: Filter datasets (create refusal / non-refusal splits) ==="
uv run -m utils.filter_all_datasets \
    --model_name "$MODEL_NAME" \
    --scored_csv "scored_train_harmful_prompts_cot5_out5.csv" \
    --cot_lower_threshold "$REFUSAL_THRESHOLD" \
    --cot_upper_threshold "$NONREFUSAL_THRESHOLD" \
    --baseline_lower_threshold "$REFUSAL_THRESHOLD" \
    --baseline_upper_threshold "$NONREFUSAL_THRESHOLD"

# Kill vLLM to free GPUs for model loading
free_gpus

# echo ""

# =============================================================================
# PHASE 2: Cache activations & create orthogonal models (needs GPU for Qwen3-8B)
# =============================================================================
echo "====== PHASE 2: CACHE ACTIVATIONS & CREATE ORTHO MODELS ======"

echo "=== Step 4: Cache activations ==="
uv run -m utils.cache_activations --model_name "$MODEL_NAME" --layers "$TRY_LAYERS" --type "$TYPE1" --refusal_threshold "$REFUSAL_THRESHOLD" --nonrefusal_threshold "$NONREFUSAL_THRESHOLD"
uv run -m utils.cache_activations --model_name "$MODEL_NAME" --layers "$TRY_LAYERS" --type "$TYPE2" --refusal_threshold "$REFUSAL_THRESHOLD" --nonrefusal_threshold "$NONREFUSAL_THRESHOLD"

echo "=== Step 5: Create orthogonal models ==="
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME" --layer "$TRY_LAYERS" --type "$TYPE1"
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME" --layer "$TRY_LAYERS" --type "$TYPE2"

echo ""

# =============================================================================
# PHASE 3: Subset generation from ortho models (needs GPU for ortho Qwen3-8B)
# =============================================================================
echo "====== PHASE 3: SUBSET GENERATION ======"

echo "=== Step 6: Batch generation (subset) ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE1" --layer "$TRY_LAYERS"
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE2" --layer "$TRY_LAYERS"

echo ""

# =============================================================================
# PHASE 4: Score subset outputs (needs vLLM server)
# =============================================================================
echo "====== PHASE 4: SCORE SUBSET OUTPUTS ======"

ensure_vllm_server

echo "=== Step 7: Compute score outputs (subset) ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --type "$TYPE1" --layers "$TRY_LAYERS" --input_dir attack_results --subset
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --type "$TYPE2" --layers "$TRY_LAYERS" --input_dir attack_results --subset

echo "=== Step 7b: Repair NaN scores (subset) ==="
for layer in ${TRY_LAYERS//,/ }; do
    for type in "$TYPE1" "$TYPE2"; do
        scored_csv="${MODEL_DIR}/attack_results/scored_ortho_output_subset_5_test_harmful_prompts_${type}_layer_${layer}.csv"
        if [ -f "$scored_csv" ]; then
            uv run -m utils.repair_scores --scored_csv "$scored_csv" --max_retries "$REPAIR_RETRIES"
        fi
    done
done

# Kill vLLM to free GPUs
free_gpus

echo ""

# =============================================================================
# PHASE 5: Layer selection (CPU only)
# =============================================================================
echo "====== PHASE 5: LAYER SELECTION ======"

echo "=== Step 8: Compute layer statistics ==="
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME" --type "$TYPE1" --layer "$TRY_LAYERS"
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME" --type "$TYPE2" --layer "$TRY_LAYERS"

echo "=== Step 9: Get best layer ==="
BEST_LAYER1=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME" --type "$TYPE1" --try_layers "$TRY_LAYERS")
echo "Best layer (cot): $BEST_LAYER1"

BEST_LAYER2=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME" --type "$TYPE2" --try_layers "$TRY_LAYERS")
echo "Best layer (baseline): $BEST_LAYER2"

echo ""

# =============================================================================
# PHASE 6: Final generation on full test set (needs GPU for ortho Qwen3-8B)
# =============================================================================
echo "====== PHASE 6: FINAL GENERATION ======"

echo "=== Step 10: Final batch generation ==="
if [ "$BEST_LAYER1" != "NONE" ]; then
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts.csv" --type "$TYPE1" --layer "$BEST_LAYER1"
else
    echo "Skipping $TYPE1 final generation - no valid layer found"
fi

if [ "$BEST_LAYER2" != "NONE" ]; then
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts.csv" --type "$TYPE2" --layer "$BEST_LAYER2"
else
    echo "Skipping $TYPE2 final generation - no valid layer found"
fi

echo ""

# =============================================================================
# PHASE 7: Score final outputs (needs vLLM server)
# =============================================================================
echo "====== PHASE 7: SCORE FINAL OUTPUTS ======"

ensure_vllm_server

echo "=== Step 11: Compute score outputs (final) ==="
if [ "$BEST_LAYER1" != "NONE" ]; then
    uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_dir attack_results \
        --input_csv "ortho_output_test_harmful_prompts_${TYPE1}_layer_${BEST_LAYER1}.csv"
else
    echo "Skipping $TYPE1 scoring - no valid layer found"
fi

if [ "$BEST_LAYER2" != "NONE" ]; then
    uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_dir attack_results \
        --input_csv "ortho_output_test_harmful_prompts_${TYPE2}_layer_${BEST_LAYER2}.csv"
else
    echo "Skipping $TYPE2 scoring - no valid layer found"
fi

echo "=== Step 11b: Repair NaN scores (final) ==="
if [ "$BEST_LAYER1" != "NONE" ]; then
    uv run -m utils.repair_scores \
        --scored_csv "${MODEL_DIR}/attack_results/scored_ortho_output_test_harmful_prompts_${TYPE1}_layer_${BEST_LAYER1}.csv" \
        --max_retries "$REPAIR_RETRIES"
fi
if [ "$BEST_LAYER2" != "NONE" ]; then
    uv run -m utils.repair_scores \
        --scored_csv "${MODEL_DIR}/attack_results/scored_ortho_output_test_harmful_prompts_${TYPE2}_layer_${BEST_LAYER2}.csv" \
        --max_retries "$REPAIR_RETRIES"
fi

echo ""

# =============================================================================
# PHASE 8: Plot comparison (CPU only)
# =============================================================================
echo "====== PHASE 8: PLOT COMPARISON ======"

echo "=== Step 12: Plot boxplot comparison ==="
if [ "$BEST_LAYER1" != "NONE" ] && [ "$BEST_LAYER2" != "NONE" ]; then
    uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME" --type "all" --layer "$BEST_LAYER1,$BEST_LAYER2"
elif [ "$BEST_LAYER1" != "NONE" ]; then
    uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME" --type "$TYPE1" --layer "$BEST_LAYER1"
elif [ "$BEST_LAYER2" != "NONE" ]; then
    uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME" --type "$TYPE2" --layer "$BEST_LAYER2"
else
    echo "No valid layers found for either type - skipping plot"
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "====== PIPELINE COMPLETE ======"
echo "Completed at: $(date)"
echo ""
echo "Results layout:"
echo "  GPT-OSS scored (NEW): ${MODEL_DIR}/"
echo "  API scored (OLD):     ${BACKUP_DIR}/"
echo ""
echo "To compare scoring methods, look at:"
echo "  ${MODEL_DIR}/figures/  (GPT-OSS scored)"
echo "  ${BACKUP_DIR}/figures/ (API scored)"

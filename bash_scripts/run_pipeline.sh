#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Generalised E2E rubric pipeline.
#
# Usage:
#   bash bash_scripts/run_pipeline.sh --model_name "Qwen/Qwen3-8B"
#   bash bash_scripts/run_pipeline.sh --model_name "Qwen/Qwen3-8B" --layers "13,15,17,19,21,23"
#   bash bash_scripts/run_pipeline.sh --model_name "Qwen/Qwen3-8B" --skip_dataset_gen
#   bash bash_scripts/run_pipeline.sh --model_name "Qwen/Qwen3-8B" --start_phase 3
#
# Full pipeline:
#   Phase 0: Base dataset creation (shared across models, only needed once)
#   Phase 1: Generate clean model outputs + score them + filter
#   Phase 2: Cache activations + create ortho models
#   Phase 3: Subset generation from ortho models
#   Phase 4: Score subset outputs + layer selection
#   Phase 5: Final generation on best layers
#   Phase 6: Score final outputs
#   Phase 7: Plot comparison
#
# Environment variables:
#   REFUSAL_THRESHOLD    - default 0.2
#   NONREFUSAL_THRESHOLD - default 0.8
#   VLLM_PORT            - default 8000
#   VLLM_TP_SIZE         - default 4
#   VLLM_MODEL           - default openai/gpt-oss-20b
# =============================================================================

# ---- Parse CLI args ----

MODEL_NAME=""
TRY_LAYERS=""
SKIP_BASE_DATASET=false
START_PHASE=1
TYPE1="cot"
TYPE2="baseline"

usage() {
    echo "Usage: $0 --model_name MODEL [--layers LAYERS] [--skip_base_dataset] [--start_phase N]"
    echo ""
    echo "  --model_name        Required. e.g. Qwen/Qwen3-8B, deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    echo "  --layers            Comma-separated layers to try. Default: 13,15,17,19,21,23"
    echo "  --skip_base_dataset Skip base dataset creation (Phase 0). Use if datasets already exist."
    echo "  --start_phase       Start from phase N (0-7). Useful for resuming. Default: 0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name)   MODEL_NAME="$2"; shift 2 ;;
        --layers)       TRY_LAYERS="$2"; shift 2 ;;
        --skip_base_dataset) SKIP_BASE_DATASET=true; shift ;;
        --start_phase)  START_PHASE="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: --model_name is required"
    usage
fi

# ---- Validate model on HF + create directory structure ----

echo "Validating model and creating directory structure..."
uv run python -c "from utils.paths import validate_hf_id, make_dirs; validate_hf_id('${MODEL_NAME}'); make_dirs('${MODEL_NAME}')"
echo "Model validated and directories ready."

# ---- Default layers per model (override with --layers) ----

if [ -z "$TRY_LAYERS" ]; then
    case "$MODEL_NAME" in
        openai/gpt-oss-20b)
            TRY_LAYERS="7,9,11,13,15,17,19" ;;
        *)
            TRY_LAYERS="13,15,17,19,21,23" ;;
    esac
fi

# ---- Config ----

REFUSAL_THRESHOLD="${REFUSAL_THRESHOLD:-0.2}"
NONREFUSAL_THRESHOLD="${NONREFUSAL_THRESHOLD:-0.8}"

RESULTS_BASE="results"
MODEL_DIR="${RESULTS_BASE}/${MODEL_NAME}"

VLLM_MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
REPAIR_RETRIES=30

# ---- Logging ----

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Sanitise model name for log filename (replace / with _)
MODEL_SLUG="${MODEL_NAME//\//_}"
LOG_FILE="${LOG_DIR}/pipeline_${MODEL_SLUG}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Pipeline started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Model: $MODEL_NAME"
echo "Layers: $TRY_LAYERS"
echo "Thresholds: refusal=${REFUSAL_THRESHOLD}, nonrefusal=${NONREFUSAL_THRESHOLD}"
echo "Scoring: ${VLLM_MODEL} via local vLLM on port ${VLLM_PORT}"
echo "Start phase: $START_PHASE"
echo "Skip base dataset: $SKIP_BASE_DATASET"
echo "========================================"
echo ""

# ---- Helper functions ----

free_gpus() {
    echo "Freeing GPUs..."

    # Kill vLLM server
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    sleep 3
    pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

    # Kill any orphaned python processes still holding GPU memory
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

# =============================================================================
# PHASE 0: Base dataset creation (shared across all models, only needed once)
# =============================================================================
if [ "$START_PHASE" -le 0 ] && [ "$SKIP_BASE_DATASET" = false ]; then
    echo "====== PHASE 0: BASE DATASET CREATION ======"

    echo "=== Step 0a: Create base datasets ==="
    uv run -m utils.create_base_dataset --dataset orbench --n 500 --dataset_dir dataset/base/
    uv run -m utils.create_base_dataset --dataset strongreject --dataset_dir dataset/base/
    uv run -m utils.create_base_dataset --dataset harmbench --dataset_dir dataset/base/
    uv run -m utils.create_base_dataset --dataset advbench --dataset_dir dataset/base/
    uv run -m utils.create_base_dataset --dataset sorrybench --dataset_dir dataset/base/

    echo "=== Step 0b: Check duplicates & create splits ==="
    uv run -m utils.check_duplicates
    uv run -m utils.create_holdout_set --train_set_split 0.75 --input_csv all_harmful_prompts.csv --seed 42
    uv run -m utils.create_random_subset --num_prompts 5 --seed 42

    echo ""
fi

# =============================================================================
# PHASE 1: Generate clean model outputs, score them, filter
# =============================================================================
if [ "$START_PHASE" -le 1 ]; then
    echo "====== PHASE 1: GENERATE & SCORE CLEAN MODEL OUTPUTS ======"

    echo "=== Step 1: Generate clean model outputs ==="
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "train_harmful_prompts.csv"
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts.csv"

    # Free GPUs before starting vLLM scorer
    free_gpus

    echo "=== Step 2: Score outputs with vLLM ==="
    ensure_vllm_server

    uv run -m utils.compute_score_outputs \
        --model_name "$MODEL_NAME" \
        --input_csv "train_harmful_prompts_cot5_out5.csv" \
        --input_dir dataset

    uv run -m utils.compute_score_outputs \
        --model_name "$MODEL_NAME" \
        --input_csv "test_harmful_prompts_cot5_out5.csv" \
        --input_dir dataset

    echo "=== Step 2b: Repair NaN scores ==="
    uv run -m utils.repair_scores \
        --scored_csv "${MODEL_DIR}/dataset/scored_train_harmful_prompts_cot5_out5.csv" \
        --max_retries "$REPAIR_RETRIES"
    uv run -m utils.repair_scores \
        --scored_csv "${MODEL_DIR}/dataset/scored_test_harmful_prompts_cot5_out5.csv" \
        --max_retries "$REPAIR_RETRIES"

    echo "=== Step 3: Filter datasets ==="
    uv run -m utils.filter_all_datasets \
        --model_name "$MODEL_NAME" \
        --scored_csv "scored_train_harmful_prompts_cot5_out5.csv" \
        --cot_lower_threshold "$REFUSAL_THRESHOLD" \
        --cot_upper_threshold "$NONREFUSAL_THRESHOLD" \
        --baseline_lower_threshold "$REFUSAL_THRESHOLD" \
        --baseline_upper_threshold "$NONREFUSAL_THRESHOLD"

    # Free GPUs for next phase
    free_gpus
    echo ""
fi

# =============================================================================
# PHASE 2: Cache activations & create ortho models (needs target model GPU)
# =============================================================================
if [ "$START_PHASE" -le 2 ]; then
    echo "====== PHASE 2: CACHE ACTIVATIONS & CREATE ORTHO MODELS ======"

    # Preflight check
    if [ ! -f "${MODEL_DIR}/dataset/scored_train_harmful_prompts_cot5_out5.csv" ]; then
        echo "ERROR: ${MODEL_DIR}/dataset/scored_train_harmful_prompts_cot5_out5.csv not found."
        echo "Run Phase 1 first."
        exit 1
    fi

    echo "=== Step 4: Cache activations ==="
    uv run -m utils.cache_activations --model_name "$MODEL_NAME" --layers "$TRY_LAYERS" --type "$TYPE1" \
        --refusal_threshold "$REFUSAL_THRESHOLD" --nonrefusal_threshold "$NONREFUSAL_THRESHOLD"
    uv run -m utils.cache_activations --model_name "$MODEL_NAME" --layers "$TRY_LAYERS" --type "$TYPE2" \
        --refusal_threshold "$REFUSAL_THRESHOLD" --nonrefusal_threshold "$NONREFUSAL_THRESHOLD"

    echo "=== Step 5: Create orthogonal models ==="
    uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME" --layer "$TRY_LAYERS" --type "$TYPE1"
    uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME" --layer "$TRY_LAYERS" --type "$TYPE2"

    echo ""
fi

# =============================================================================
# PHASE 3: Subset generation from ortho models (needs target model GPU)
# =============================================================================
if [ "$START_PHASE" -le 3 ]; then
    echo "====== PHASE 3: SUBSET GENERATION ======"

    echo "=== Step 6: Batch generation (subset) ==="
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE1" --layer "$TRY_LAYERS"
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE2" --layer "$TRY_LAYERS"

    echo ""
fi

# =============================================================================
# PHASE 4: Score subset outputs + layer selection
# =============================================================================
if [ "$START_PHASE" -le 4 ]; then
    echo "====== PHASE 4: SCORE SUBSET & SELECT BEST LAYERS ======"

    # Free GPUs and start vLLM for scoring
    free_gpus
    ensure_vllm_server

    echo "=== Step 7: Score subset outputs ==="
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

    echo "=== Step 8: Compute layer statistics ==="
    uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME" --type "$TYPE1" --layer "$TRY_LAYERS"
    uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME" --type "$TYPE2" --layer "$TRY_LAYERS"

    echo "=== Step 9: Get best layers ==="
    BEST_LAYER1=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME" --type "$TYPE1" --try_layers "$TRY_LAYERS")
    echo "Best layer (${TYPE1}): $BEST_LAYER1"

    BEST_LAYER2=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME" --type "$TYPE2" --try_layers "$TRY_LAYERS")
    echo "Best layer (${TYPE2}): $BEST_LAYER2"

    # Save best layers to file for resume support
    mkdir -p "$MODEL_DIR"
    echo "$BEST_LAYER1" > "${MODEL_DIR}/.best_layer_${TYPE1}"
    echo "$BEST_LAYER2" > "${MODEL_DIR}/.best_layer_${TYPE2}"

    # Free GPUs for next phase
    free_gpus
    echo ""
fi

# =============================================================================
# PHASE 5: Final generation on full test set (needs target model GPU)
# =============================================================================
if [ "$START_PHASE" -le 5 ]; then
    echo "====== PHASE 5: FINAL GENERATION ======"

    # Load best layers (either from phase 4 or from saved file)
    if [ -z "${BEST_LAYER1:-}" ]; then
        BEST_LAYER1=$(cat "${MODEL_DIR}/.best_layer_${TYPE1}" 2>/dev/null || echo "NONE")
    fi
    if [ -z "${BEST_LAYER2:-}" ]; then
        BEST_LAYER2=$(cat "${MODEL_DIR}/.best_layer_${TYPE2}" 2>/dev/null || echo "NONE")
    fi

    echo "Using best layers - ${TYPE1}: $BEST_LAYER1, ${TYPE2}: $BEST_LAYER2"

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
fi

# =============================================================================
# PHASE 6: Score final outputs (needs vLLM server)
# =============================================================================
if [ "$START_PHASE" -le 6 ]; then
    echo "====== PHASE 6: SCORE FINAL OUTPUTS ======"

    # Load best layers
    if [ -z "${BEST_LAYER1:-}" ]; then
        BEST_LAYER1=$(cat "${MODEL_DIR}/.best_layer_${TYPE1}" 2>/dev/null || echo "NONE")
    fi
    if [ -z "${BEST_LAYER2:-}" ]; then
        BEST_LAYER2=$(cat "${MODEL_DIR}/.best_layer_${TYPE2}" 2>/dev/null || echo "NONE")
    fi

    # Free GPUs and start vLLM
    free_gpus
    ensure_vllm_server

    echo "=== Step 11: Score final outputs ==="
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
fi

# =============================================================================
# PHASE 7: Plot comparison
# =============================================================================
if [ "$START_PHASE" -le 7 ]; then
    echo "====== PHASE 7: PLOT COMPARISON ======"

    # Load best layers
    if [ -z "${BEST_LAYER1:-}" ]; then
        BEST_LAYER1=$(cat "${MODEL_DIR}/.best_layer_${TYPE1}" 2>/dev/null || echo "NONE")
    fi
    if [ -z "${BEST_LAYER2:-}" ]; then
        BEST_LAYER2=$(cat "${MODEL_DIR}/.best_layer_${TYPE2}" 2>/dev/null || echo "NONE")
    fi

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
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "====== PIPELINE COMPLETE ======"
echo "Completed at: $(date)"
echo ""
echo "Results: ${MODEL_DIR}/"
echo "Log:     ${LOG_FILE}"

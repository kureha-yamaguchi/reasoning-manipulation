#!/bin/bash
# Score GCG+IRIS attack results: generate rollouts + StrongREJECT scoring.
#
# Phase 1: Generate N rollouts per attacked prompt using vLLM (target model)
# Phase 2: Score rollouts with StrongREJECT via local vLLM (judge model)
# Phase 2b: Repair any NaN scores
#
# Usage:
#   bash bash_scripts/run_attack_scoring.sh                    # Full pipeline
#   bash bash_scripts/run_attack_scoring.sh --start_phase 2    # Resume from phase 2
#   bash bash_scripts/run_attack_scoring.sh --generate-only    # Phase 1 only
#   bash bash_scripts/run_attack_scoring.sh --score-only       # Phase 2 only
#   bash bash_scripts/run_attack_scoring.sh --counterfactual-only  # Extract baselines only
#   bash bash_scripts/run_attack_scoring.sh --run-tag v2           # Use run tag

set -euo pipefail

START_PHASE=1
MODE=""
RUN_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start_phase)  START_PHASE="$2"; shift 2 ;;
        --run-tag)      RUN_TAG="$2"; shift 2 ;;
        --generate-only|--score-only|--counterfactual-only) MODE="$1"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
NUM_ROLLOUTS=5
REFUSAL_MODE="cot"

MODELS=("deepseek-llama-8b" "deepseek-qwen-7b" "qwen3-8b" "gpt-oss-20b")

VLLM_MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
REPAIR_RETRIES="${REPAIR_RETRIES:-30}"
TAG_SUFFIX=""
TAG_FLAG=""
if [ -n "$RUN_TAG" ]; then
    TAG_SUFFIX="_${RUN_TAG}"
    TAG_FLAG="--run-tag $RUN_TAG"
fi

BETAS=("0p0" "0p3" "0p5" "0p7" "1p0")

LOG_DIR="prompt_attack/logs"
mkdir -p "$LOG_DIR"

echo "=== GCG+IRIS Attack Scoring Pipeline ==="
echo "Models: ${MODELS[*]}"
echo "Rollouts: $NUM_ROLLOUTS, Mode: $REFUSAL_MODE"
echo "Run tag: ${RUN_TAG:-<none>}"
echo "Start phase: $START_PHASE"
echo "========================================="

# ---- Helper functions ----

start_vllm_judge() {
    echo "Starting vLLM judge server for ${VLLM_MODEL}..."
    VLLM_LOG="${LOG_DIR}/vllm_judge_$(date +%Y%m%d_%H%M%S).log"
    uv run python -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID (log: $VLLM_LOG)"

    # Wait for server to be ready
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "  vLLM judge server ready."
            return 0
        fi
        sleep 5
    done
    echo "  ERROR: vLLM judge server failed to start within 5 minutes."
    kill $VLLM_PID 2>/dev/null || true
    return 1
}

ensure_vllm_judge() {
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM judge server already running."
    else
        start_vllm_judge
    fi
    export STRONGREJECT_VLLM_URL="$VLLM_URL"
}

# ---- Phase 1: Generate rollouts ----

generate_rollouts() {
    echo ""
    echo "====== PHASE 1: GENERATE ROLLOUTS ======"
    for model in "${MODELS[@]}"; do
        log_file="${LOG_DIR}/rollouts_${model}_${REFUSAL_MODE}.log"
        echo ""
        echo ">>> Generating rollouts: model=$model"
        echo "    Log: $log_file"

        CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m prompt_attack.generate_rollouts \
            --model "$model" \
            --num-rollouts "$NUM_ROLLOUTS" \
            --refusal-mode "$REFUSAL_MODE" \
            $TAG_FLAG \
            2>&1 | tee "$log_file"

        echo "    Done: $model"
    done
}

# ---- Phase 2: Score rollouts ----

score_rollouts() {
    echo ""
    echo "====== PHASE 2: SCORE ROLLOUTS ======"
    ensure_vllm_judge

    for model in "${MODELS[@]}"; do
        log_file="${LOG_DIR}/scoring_${model}_${REFUSAL_MODE}.log"
        echo ""
        echo ">>> Scoring rollouts: model=$model"
        echo "    Log: $log_file"

        uv run python -m prompt_attack.score_rollouts \
            --model "$model" \
            --refusal-mode "$REFUSAL_MODE" \
            $TAG_FLAG \
            2>&1 | tee "$log_file"

        echo "    Done: $model"
    done

    # Repair NaN scores
    echo ""
    echo "=== Phase 2b: Repair NaN scores ==="
    for model in "${MODELS[@]}"; do
        mc_subdir=$(uv run python -c "from prompt_attack.models import get_model_config; print(get_model_config('$model').results_subdir)")
        for beta in "${BETAS[@]}"; do
            scored_csv="results/${mc_subdir}/prompt_attack/scored_rollouts_${model}_beta_${beta}_${REFUSAL_MODE}${TAG_SUFFIX}.csv"
            if [ -f "$scored_csv" ]; then
                echo ">>> Repairing: $scored_csv"
                uv run python -m utils.repair_scores \
                    --scored_csv "$scored_csv" \
                    --max_retries "$REPAIR_RETRIES"
            fi
        done
    done
}

# ---- Extract counterfactual only ----

extract_counterfactual() {
    echo ""
    echo "====== EXTRACTING COUNTERFACTUAL SCORES ======"
    for model in "${MODELS[@]}"; do
        echo ">>> Extracting counterfactual: model=$model"
        uv run python -m prompt_attack.score_rollouts \
            --model "$model" \
            --refusal-mode "$REFUSAL_MODE" \
            $TAG_FLAG \
            --counterfactual-only
    done
}

# ---- Main ----

case "$MODE" in
    --generate-only)
        generate_rollouts
        ;;
    --score-only)
        score_rollouts
        ;;
    --counterfactual-only)
        extract_counterfactual
        ;;
    *)
        if [ "$START_PHASE" -le 1 ]; then
            generate_rollouts
        fi
        if [ "$START_PHASE" -le 2 ]; then
            score_rollouts
        fi
        ;;
esac

echo ""
echo "=== Scoring pipeline complete ==="

#!/bin/bash
# Run GCG+IRIS attacks across all models and beta values.
# Results go into results/<org>/<model>/prompt_attack/
# Usage:
#   bash bash_scripts/run_attacks.sh                         # Full run
#   bash bash_scripts/run_attacks.sh --run-tag v2            # Separate from previous runs
#   bash bash_scripts/run_attacks.sh --dry-run               # Preview only
#   bash bash_scripts/run_attacks.sh --run-tag v2 --dry-run  # Both

set -euo pipefail

RUN_TAG=""
DRY_RUN=""
REFUSED_ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-tag)       RUN_TAG="$2"; shift 2 ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --refused-only)  REFUSED_ONLY="--refused-only"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

NUM_GPUS=4
NUM_STEPS=500
NUM_PROMPTS=10
REFUSAL_MODE="cot"
TARGET_TOKENS=20
SEARCH_WIDTH=512

MODELS=("deepseek-llama-8b" "deepseek-qwen-7b" "qwen3-8b" "gpt-oss-20b")
# BETAS=("0.25" "0.50" "0.75")
BETAS=("0.75" "0.50" "0.25")

LOG_DIR="prompt_attack/logs"
mkdir -p "$LOG_DIR"

echo "=== GCG+IRIS Attack Sweep ==="
echo "Models: ${MODELS[*]}"
echo "Betas: ${BETAS[*]}"
echo "Steps: $NUM_STEPS, Prompts: $NUM_PROMPTS, GPUs: $NUM_GPUS"
[ -n "$REFUSED_ONLY" ] && echo "Prompt selection: refused-only"
[ -n "$RUN_TAG" ] && echo "Run tag: $RUN_TAG"
echo "=============================="

for model in "${MODELS[@]}"; do
    # gpt-oss is 20B MoE — reduce search width to avoid slow batches
    if [ "$model" = "gpt-oss-20b" ]; then
        model_search_width=128
    else
        model_search_width=$SEARCH_WIDTH
    fi

    for beta in "${BETAS[@]}"; do
        tag_suffix=""
        [ -n "$RUN_TAG" ] && tag_suffix="_${RUN_TAG}"
        log_file="${LOG_DIR}/${model}_beta${beta}_${REFUSAL_MODE}${tag_suffix}.log"
        echo ""
        echo ">>> Running: model=$model beta=$beta mode=$REFUSAL_MODE search_width=$model_search_width"
        echo "    Log: $log_file"

        CMD="uv run python -m prompt_attack.run_attack \
            --model $model \
            --beta $beta \
            --num-gpus $NUM_GPUS \
            --num-steps $NUM_STEPS \
            --refusal-mode $REFUSAL_MODE \
            --target-tokens $TARGET_TOKENS \
            --search-width $model_search_width \
            --num-prompts $NUM_PROMPTS \
            --use-common-prompts \
            $REFUSED_ONLY"

        [ -n "$RUN_TAG" ] && CMD="$CMD --run-tag $RUN_TAG"
        [ -n "$DRY_RUN" ] && CMD="$CMD --dry-run"

        echo "    Command: $CMD"
        eval "$CMD" 2>&1 | tee "$log_file"

        echo "    Done: model=$model beta=$beta"
    done
done

echo ""
echo "=== All attacks complete ==="

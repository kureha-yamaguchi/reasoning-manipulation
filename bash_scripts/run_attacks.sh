#!/bin/bash
# Run GCG+IRIS attacks across all models and beta values.
# Results go into results/<org>/<model>/prompt_attack/
# Usage: bash bash_scripts/run_attacks.sh [--dry-run]

set -euo pipefail

DRY_RUN="${1:-}"
NUM_GPUS=4
NUM_STEPS=150
NUM_PROMPTS=50
REFUSAL_MODE="cot"
TARGET_TOKENS=20

MODELS=("deepseek-llama-8b" "deepseek-qwen-7b" "qwen3-8b" "gpt-oss-20b")
BETAS=("0.0" "0.3" "0.5" "0.7" "1.0")

LOG_DIR="prompt_attack/logs"
mkdir -p "$LOG_DIR"

echo "=== GCG+IRIS Attack Sweep ==="
echo "Models: ${MODELS[*]}"
echo "Betas: ${BETAS[*]}"
echo "Steps: $NUM_STEPS, Prompts: $NUM_PROMPTS, GPUs: $NUM_GPUS"
echo "=============================="

for model in "${MODELS[@]}"; do
    for beta in "${BETAS[@]}"; do
        log_file="${LOG_DIR}/${model}_beta${beta}_${REFUSAL_MODE}.log"
        echo ""
        echo ">>> Running: model=$model beta=$beta mode=$REFUSAL_MODE"
        echo "    Log: $log_file"

        CMD="uv run python -m prompt_attack.run_attack \
            --model $model \
            --beta $beta \
            --num-gpus $NUM_GPUS \
            --num-steps $NUM_STEPS \
            --refusal-mode $REFUSAL_MODE \
            --target-tokens $TARGET_TOKENS \
            --num-prompts $NUM_PROMPTS \
            --use-common-prompts"

        if [ "$DRY_RUN" = "--dry-run" ]; then
            CMD="$CMD --dry-run"
        fi

        echo "    Command: $CMD"
        eval "$CMD" 2>&1 | tee "$log_file"

        echo "    Done: model=$model beta=$beta"
    done
done

echo ""
echo "=== All attacks complete ==="

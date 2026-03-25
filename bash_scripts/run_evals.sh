#!/usr/bin/env bash
set -euo pipefail

# Off-task eval pipeline: run Inspect evals on base vs ortho models
# Uses vLLM to serve each model, runs AIME 2025, GPQA Diamond, and MATH

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_pipeline_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Off-Task Eval Pipeline"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "========================================"
echo ""

# # ── Install dependencies ──────────────────────────────────────────────
# echo "=== Installing inspect-ai and inspect-evals ==="
# uv pip install inspect-ai "inspect-evals[aime2025,gpqa,mathematics]"
# echo ""

# ── Common settings ───────────────────────────────────────────────────
PORT=8001
BENCHMARKS="aime2025 gpqa_diamond math"
TEMPERATURE=0.6
MAX_TOKENS=32768
TP_SIZE=4

# ── Model configurations ─────────────────────────────────────────────
# Each entry: MODEL_PATH|MODEL_LABEL
declare -a MODELS=(
    # ── Ortho models (best layers from .best_layer_* files) ──
    # Qwen3-8B: cot=23, baseline=17
    "${PROJECT_ROOT}/results/Qwen/Qwen3-8B/ortho_model_cot_layer_23|Qwen/Qwen3-8B/ortho_model_cot_layer_23"
    "${PROJECT_ROOT}/results/Qwen/Qwen3-8B/ortho_model_baseline_layer_17|Qwen/Qwen3-8B/ortho_model_baseline_layer_17"
    # gpt-oss-20b: cot=19, baseline=15
    "${PROJECT_ROOT}/results/openai/gpt-oss-20b/ortho_model_cot_layer_19|openai/gpt-oss-20b/ortho_model_cot_layer_19"
    "${PROJECT_ROOT}/results/openai/gpt-oss-20b/ortho_model_baseline_layer_15|openai/gpt-oss-20b/ortho_model_baseline_layer_15"
    # DeepSeek-R1-Distill-Llama-8B: cot=23, baseline=15
    "${PROJECT_ROOT}/results/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/ortho_model_cot_layer_23|deepseek-ai/DeepSeek-R1-Distill-Llama-8B/ortho_model_cot_layer_23"
    "${PROJECT_ROOT}/results/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/ortho_model_baseline_layer_15|deepseek-ai/DeepSeek-R1-Distill-Llama-8B/ortho_model_baseline_layer_15"
    # DeepSeek-R1-Distill-Qwen-7B: cot=21, baseline=17
    "${PROJECT_ROOT}/results/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ortho_model_cot_layer_21|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ortho_model_cot_layer_21"
    "${PROJECT_ROOT}/results/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ortho_model_baseline_layer_17|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ortho_model_baseline_layer_17"
)

# ── Run evals for each model ─────────────────────────────────────────
for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_PATH MODEL_LABEL <<< "$entry"

    echo ""
    echo "========================================================"
    echo "Evaluating: $MODEL_LABEL"
    echo "  Model path: $MODEL_PATH"
    echo "========================================================"
    echo ""

    uv run python "${PROJECT_ROOT}/evals/run_inspect_evals.py" \
        --model_path "$MODEL_PATH" \
        --model_label "$MODEL_LABEL" \
        --benchmarks $BENCHMARKS \
        --port "$PORT" \
        --tensor_parallel_size "$TP_SIZE" \
        --gpu_memory_utilization 0.9 \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS"

    echo ""
    echo "Completed: $MODEL_LABEL"
    echo ""
done

# ── Generate summary table ────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Generating summary table"
echo "========================================================"
echo ""

uv run python "${PROJECT_ROOT}/evals/summarise_results.py" \
    --csv "${PROJECT_ROOT}/results/eval_summary.csv"

echo ""
echo "========================================"
echo "Pipeline completed at: $(date)"
echo "========================================"

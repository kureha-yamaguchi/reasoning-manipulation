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
echo "=== Installing inspect-ai and inspect-evals ==="
uv pip install inspect-ai "inspect-evals[aime2025,gpqa,mathematics, frontiermath, bbeh]"
echo ""

# ── Common settings ───────────────────────────────────────────────────
PORT=8001
declare -a ALL_BENCHMARKS=("aime2025" "gpqa_diamond" "bbeh_mini")
TEMPERATURE=0.6
MAX_TOKENS=32768
TP_SIZE=4

# ── Model configurations ─────────────────────────────────────────────
# Each entry: MODEL_PATH|MODEL_LABEL
# Auto-picks best ortho CoT layer from .best_layer_cot files
declare -a MODEL_NAMES=(
    "Qwen/Qwen3-8B"
    "openai/gpt-oss-20b"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

declare -a ALL_MODELS=()

# basline models options
for model_name in "${MODEL_NAMES[@]}"; do
    ALL_MODELS+=("${model_name}|${model_name}")
done


# ortho models options
for model_name in "${MODEL_NAMES[@]}"; do
    best_layer_file="${PROJECT_ROOT}/results/${model_name}/.best_layer_cot"
    if [[ ! -f "$best_layer_file" ]]; then
        echo "ERROR: Missing .best_layer_cot for ${model_name}" >&2
        exit 1
    fi
    best_layer=$(cat "$best_layer_file")
    ALL_MODELS+=("${PROJECT_ROOT}/results/${model_name}/ortho_model_cot_layer_${best_layer}|${model_name}/ortho_model_cot_layer_${best_layer}")
done

# ── Interactive model selection ──────────────────────────────────────
echo "Available models (all selected by default):"
declare -a SELECTED=()
for i in "${!ALL_MODELS[@]}"; do
    SELECTED+=("1")
    IFS='|' read -r _ label <<< "${ALL_MODELS[$i]}"
    echo "  [$((i+1))] [x] $label"
done

echo ""
echo "Enter numbers to toggle off/on, or press Enter to continue:"
while true; do
    read -rp "> " input
    [[ -z "$input" ]] && break
    for num in $input; do
        idx=$((num - 1))
        if (( idx >= 0 && idx < ${#ALL_MODELS[@]} )); then
            if [[ "${SELECTED[$idx]}" == "1" ]]; then
                SELECTED[$idx]="0"
            else
                SELECTED[$idx]="1"
            fi
        else
            echo "  Invalid number: $num"
        fi
    done
    # Re-display current selection
    echo ""
    for i in "${!ALL_MODELS[@]}"; do
        IFS='|' read -r _ label <<< "${ALL_MODELS[$i]}"
        if [[ "${SELECTED[$i]}" == "1" ]]; then
            echo "  [$((i+1))] [x] $label"
        else
            echo "  [$((i+1))] [ ] $label"
        fi
    done
    echo ""
    echo "Enter numbers to toggle, or press Enter to continue:"
done

declare -a MODELS=()
for i in "${!ALL_MODELS[@]}"; do
    if [[ "${SELECTED[$i]}" == "1" ]]; then
        MODELS+=("${ALL_MODELS[$i]}")
    fi
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No models selected. Exiting."
    exit 0
fi

echo ""

# ── Interactive benchmark selection ──────────────────────────────────
echo "Available benchmarks (all selected by default):"
declare -a BENCH_SELECTED=()
for i in "${!ALL_BENCHMARKS[@]}"; do
    BENCH_SELECTED+=("1")
    echo "  [$((i+1))] [x] ${ALL_BENCHMARKS[$i]}"
done

echo ""
echo "Enter numbers to toggle off/on, or press Enter to continue:"
while true; do
    read -rp "> " input
    [[ -z "$input" ]] && break
    for num in $input; do
        idx=$((num - 1))
        if (( idx >= 0 && idx < ${#ALL_BENCHMARKS[@]} )); then
            if [[ "${BENCH_SELECTED[$idx]}" == "1" ]]; then
                BENCH_SELECTED[$idx]="0"
            else
                BENCH_SELECTED[$idx]="1"
            fi
        else
            echo "  Invalid number: $num"
        fi
    done
    echo ""
    for i in "${!ALL_BENCHMARKS[@]}"; do
        if [[ "${BENCH_SELECTED[$i]}" == "1" ]]; then
            echo "  [$((i+1))] [x] ${ALL_BENCHMARKS[$i]}"
        else
            echo "  [$((i+1))] [ ] ${ALL_BENCHMARKS[$i]}"
        fi
    done
    echo ""
    echo "Enter numbers to toggle, or press Enter to continue:"
done

BENCHMARKS=""
for i in "${!ALL_BENCHMARKS[@]}"; do
    if [[ "${BENCH_SELECTED[$i]}" == "1" ]]; then
        BENCHMARKS="${BENCHMARKS} ${ALL_BENCHMARKS[$i]}"
    fi
done
BENCHMARKS="${BENCHMARKS# }"  # trim leading space

if [[ -z "$BENCHMARKS" ]]; then
    echo "No benchmarks selected. Exiting."
    exit 0
fi

echo ""
echo "Running ${#MODELS[@]} model(s) x $(echo $BENCHMARKS | wc -w) benchmark(s)."

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

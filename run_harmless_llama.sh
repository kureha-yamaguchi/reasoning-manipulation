#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TYPE="cot"
TRY_LAYERS="16,17,18,19"
BEST_LAYER="17"

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Model: $MODEL_NAME | Type: $TYPE | Layers: $TRY_LAYERS"
echo ""


echo "=== Step 7: Final batch generation ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmless_prompts.csv" 
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmless_prompts.csv" --type "$TYPE" --layer "$BEST_LAYER"

echo "=== Step 8: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_csv test_harmless_prompts_cot5_out5.csv
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --type "$TYPE" --layers "$BEST_LAYER" --is_harmless

echo "=== Step 9: Plot boxplot comparison ==="
uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME" --type "$TYPE" --layer "$BEST_LAYER"
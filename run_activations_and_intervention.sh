#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="openai/gpt-oss-20b"
TYPE="baseline"
TRY_LAYERS="7,9,11,13,15,17,19 "

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

echo "=== Step 1: Cache activations ==="
uv run -m utils.cache_activations --model_name "$MODEL_NAME" --layers "$TRY_LAYERS" --type "$TYPE"

echo "=== Step 2: Create orthogonal model ==="
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME" --layer "$TRY_LAYERS" --type "$TYPE"

echo "=== Step 3: Batch generation (subset) ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE" --layer "$TRY_LAYERS"

echo "=== Step 4: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --type "$TYPE" --layers "$TRY_LAYERS" --input_dir attack_results --subset

echo "=== Step 5: Compute layer statistics ==="
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME" --type "$TYPE" --layer "$TRY_LAYERS"

echo "=== Step 6: Get best layer ==="
BEST_LAYER=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME" --type "$TYPE" --try_layers "$TRY_LAYERS")
echo "Best layer: $BEST_LAYER"

echo "=== Step 7: Final batch generation ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts.csv" --type "$TYPE" --layer "$BEST_LAYER"

echo "=== Step 8: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE}_layer_${BEST_LAYER}.csv"

echo "=== Step 9: Plot boxplot comparison ==="
uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME" --type "$TYPE" --layer "$BEST_LAYER"
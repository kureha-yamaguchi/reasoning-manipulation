#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3-8B"
TYPE1="cot"
TYPE2="baseline"
TRY_LAYERS="13,15,17,19,21,23"

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Model: $MODEL_NAME  | Type1: $TYPE1 | Type2: $TYPE2 | Layers: $TRY_LAYERS"
echo ""

echo "=== Step 5: Create orthogonal model (harmless) ==="
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME" --layer "$TRY_LAYERS" --type "$TYPE1" --harmless
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME" --layer "$TRY_LAYERS" --type "$TYPE2" --harmless

echo "=== Step 6: Batch generation subset (harmless) ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE1" --layer "$TRY_LAYERS" --harmless
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE2" --layer "$TRY_LAYERS" --harmless

echo "=== Step 7: Compute score outputs (harmless) ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --type "$TYPE1" --layers "$TRY_LAYERS" --input_dir attack_results --subset --harmless
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --type "$TYPE2" --layers "$TRY_LAYERS" --input_dir attack_results --subset --harmless

echo "=== Step 8: Compute layer statistics (harmless) ==="
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME" --type "$TYPE1" --layer "$TRY_LAYERS" --harmless
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME" --type "$TYPE2" --layer "$TRY_LAYERS" --harmless

echo "=== Step 9: Get best layer (harmless) ==="
BEST_LAYER1=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME" --type "$TYPE1" --try_layers "$TRY_LAYERS" --harmless)
echo "Best layer (cot, harmless): $BEST_LAYER1"

BEST_LAYER2=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME" --type "$TYPE2" --try_layers "$TRY_LAYERS" --harmless)
echo "Best layer (baseline, harmless): $BEST_LAYER2"

echo "=== Step 10: Final batch generation (harmless) ==="
if [ "$BEST_LAYER1" != "NONE" ]; then
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts.csv" --type "$TYPE1" --layer "$BEST_LAYER1" --harmless
else
    echo "Skipping $TYPE1 final generation - no valid layer found"
fi

if [ "$BEST_LAYER2" != "NONE" ]; then
    uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts.csv" --type "$TYPE2" --layer "$BEST_LAYER2" --harmless
else
    echo "Skipping $TYPE2 final generation - no valid layer found"
fi

echo "=== Step 11: Compute score outputs (harmless) ==="
if [ "$BEST_LAYER1" != "NONE" ]; then
    uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE1}_layer_${BEST_LAYER1}_harmless.csv"
else
    echo "Skipping $TYPE1 scoring - no valid layer found"
fi

if [ "$BEST_LAYER2" != "NONE" ]; then
    uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE2}_layer_${BEST_LAYER2}_harmless.csv"
else
    echo "Skipping $TYPE2 scoring - no valid layer found"
fi

echo "Finished at: $(date)"

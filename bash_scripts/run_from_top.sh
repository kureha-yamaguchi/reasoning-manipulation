#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME1="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME2="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME3="Qwen/Qwen3-8B"
TYPE1="cot"
TYPE2="baseline"
TRY_LAYERS="13,15,17,19,21,23"

uv pip install git+https://github.com/b-d-e/strong_reject

# Verify the installation is correct
echo "=== Verifying strong_reject installation ==="
uv pip show strong-reject | grep -E "(Name|Version|Location)"

# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Model: $MODEL_NAME1 | Type1: $TYPE1 | Type2: $TYPE2 | Layers: $TRY_LAYERS"
echo ""

echo "====== SECTION 1: CREATE DATASET ======"

echo "=== Step 1: Generate clean model datasets ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME1" --input_csv "train_harmful_prompts.csv" 
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME1" --input_csv "test_harmful_prompts.csv" 

echo "=== Step 2: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME1" --input_csv "train_harmful_prompts_cot5_out5.csv" --input_dir dataset
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME1" --input_csv "test_harmful_prompts_cot5_out5.csv" --input_dir dataset

echo "=== Step 3: Create refusal and non-refusal datasets (both cot and baseline) ==="
uv run -m utils.filter_all_datasets --model_name "$MODEL_NAME1" --scored_csv "scored_train_harmful_prompts_cot5_out5.csv"

echo "====== SECTION 2: CACHE ACTIVATIONS AND CREATE ORTHOGONAL MODELS ======"

echo "=== Step 4: Cache activations ==="
uv run -m utils.cache_activations --model_name "$MODEL_NAME1" --layers "$TRY_LAYERS" --type "$TYPE1"

uv run -m utils.cache_activations --model_name "$MODEL_NAME1" --layers "$TRY_LAYERS" --type "$TYPE2"


echo "=== Step 5: Create orthogonal model ==="
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME1" --layer "$TRY_LAYERS" --type "$TYPE1"

uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME1" --layer "$TRY_LAYERS" --type "$TYPE2"

echo "=== Step 6: Batch generation (subset) ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME1" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE1" --layer "$TRY_LAYERS"

uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME1" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE2" --layer "$TRY_LAYERS"

echo "=== Step 7: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME1" --type "$TYPE1" --layers "$TRY_LAYERS" --input_dir attack_results --subset

uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME1" --type "$TYPE2" --layers "$TRY_LAYERS" --input_dir attack_results --subset

echo "=== Step 8: Compute layer statistics ==="
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME1" --type "$TYPE1" --layer "$TRY_LAYERS"

uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME1" --type "$TYPE2" --layer "$TRY_LAYERS"

echo "=== Step 9: Get best layer ==="
BEST_LAYER1=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME1" --type "$TYPE1" --try_layers "$TRY_LAYERS")
echo "Best layer (cot): $BEST_LAYER1"

BEST_LAYER2=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME1" --type "$TYPE2" --try_layers "$TRY_LAYERS")
echo "Best layer (baseline): $BEST_LAYER2"

echo "=== Step 10: Final batch generation ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME1" --input_csv "test_harmful_prompts.csv" --type "$TYPE1" --layer "$BEST_LAYER1"

uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME1" --input_csv "test_harmful_prompts.csv" --type "$TYPE2" --layer "$BEST_LAYER2"

echo "=== Step 11: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME1" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE1}_layer_${BEST_LAYER1}.csv"

uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME1" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE2}_layer_${BEST_LAYER2}.csv"

echo "=== Step 12: Plot boxplot comparison ==="
uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME1" --type "all" --layer "$BEST_LAYER1,$BEST_LAYER2"


echo "Model: $MODEL_NAME2 | Type1: $TYPE1 | Type2: $TYPE2 | Layers: $TRY_LAYERS"
echo ""

echo "====== SECTION 1: CREATE DATASET ======"

echo "=== Step 1: Generate clean model datasets ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME2" --input_csv "train_harmful_prompts.csv" 
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME2" --input_csv "test_harmful_prompts.csv" 

echo "=== Step 2: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME2" --input_csv "train_harmful_prompts_cot5_out5.csv" --input_dir dataset
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME2" --input_csv "test_harmful_prompts_cot5_out5.csv" --input_dir dataset

echo "=== Step 3: Create refusal and non-refusal datasets (both cot and baseline) ==="
uv run -m utils.filter_all_datasets --model_name "$MODEL_NAME2" --scored_csv "scored_train_harmful_prompts_cot5_out5.csv"

echo "====== SECTION 2: CACHE ACTIVATIONS AND CREATE ORTHOGONAL MODELS ======"

echo "=== Step 4: Cache activations ==="
uv run -m utils.cache_activations --model_name "$MODEL_NAME2" --layers "$TRY_LAYERS" --type "$TYPE1"

uv run -m utils.cache_activations --model_name "$MODEL_NAME2" --layers "$TRY_LAYERS" --type "$TYPE2"


echo "=== Step 5: Create orthogonal model ==="
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME2" --layer "$TRY_LAYERS" --type "$TYPE1"

uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME2" --layer "$TRY_LAYERS" --type "$TYPE2"

echo "=== Step 6: Batch generation (subset) ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME2" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE1" --layer "$TRY_LAYERS"

uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME2" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE2" --layer "$TRY_LAYERS"

echo "=== Step 7: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME2" --type "$TYPE1" --layers "$TRY_LAYERS" --input_dir attack_results --subset

uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME2" --type "$TYPE2" --layers "$TRY_LAYERS" --input_dir attack_results --subset

echo "=== Step 8: Compute layer statistics ==="
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME2" --type "$TYPE1" --layer "$TRY_LAYERS"

uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME2" --type "$TYPE2" --layer "$TRY_LAYERS"

echo "=== Step 9: Get best layer ==="
BEST_LAYER1=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME2" --type "$TYPE1" --try_layers "$TRY_LAYERS")
echo "Best layer (cot): $BEST_LAYER1"

BEST_LAYER2=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME2" --type "$TYPE2" --try_layers "$TRY_LAYERS")
echo "Best layer (baseline): $BEST_LAYER2"

echo "=== Step 10: Final batch generation ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME2" --input_csv "test_harmful_prompts.csv" --type "$TYPE1" --layer "$BEST_LAYER1"

uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME2" --input_csv "test_harmful_prompts.csv" --type "$TYPE2" --layer "$BEST_LAYER2"

echo "=== Step 11: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME2" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE1}_layer_${BEST_LAYER1}.csv"

uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME2" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE2}_layer_${BEST_LAYER2}.csv"

echo "=== Step 12: Plot boxplot comparison ==="
uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME2" --type "all" --layer "$BEST_LAYER1,$BEST_LAYER2"


echo "Model: $MODEL_NAME3 | Type1: $TYPE1 | Type2: $TYPE2 | Layers: $TRY_LAYERS"
echo ""

echo "====== SECTION 1: CREATE DATASET ======"

echo "=== Step 1: Generate clean model datasets ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME3" --input_csv "train_harmful_prompts.csv" 
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME3" --input_csv "test_harmful_prompts.csv" 

echo "=== Step 2: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME3" --input_csv "train_harmful_prompts_cot5_out5.csv" --input_dir dataset
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME3" --input_csv "test_harmful_prompts_cot5_out5.csv" --input_dir dataset

echo "=== Step 3: Create refusal and non-refusal datasets (both cot and baseline) ==="
uv run -m utils.filter_all_datasets --model_name "$MODEL_NAME3" --scored_csv "scored_train_harmful_prompts_cot5_out5.csv"

echo "====== SECTION 2: CACHE ACTIVATIONS AND CREATE ORTHOGONAL MODELS ======"

echo "=== Step 4: Cache activations ==="
uv run -m utils.cache_activations --model_name "$MODEL_NAME3" --layers "$TRY_LAYERS" --type "$TYPE1"

uv run -m utils.cache_activations --model_name "$MODEL_NAME3" --layers "$TRY_LAYERS" --type "$TYPE2"


echo "=== Step 5: Create orthogonal model ==="
uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME3" --layer "$TRY_LAYERS" --type "$TYPE1"

uv run -m interventions.create_ortho_model --model_name "$MODEL_NAME3" --layer "$TRY_LAYERS" --type "$TYPE2"

echo "=== Step 6: Batch generation (subset) ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME3" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE1" --layer "$TRY_LAYERS"

uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME3" --input_csv "subset_5_test_harmful_prompts.csv" --type "$TYPE2" --layer "$TRY_LAYERS"

echo "=== Step 7: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME3" --type "$TYPE1" --layers "$TRY_LAYERS" --input_dir attack_results --subset

uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME3" --type "$TYPE2" --layers "$TRY_LAYERS" --input_dir attack_results --subset

echo "=== Step 8: Compute layer statistics ==="
uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME3" --type "$TYPE1" --layer "$TRY_LAYERS"

uv run -m utils.compute_layer_statistics --model_name "$MODEL_NAME3" --type "$TYPE2" --layer "$TRY_LAYERS"

echo "=== Step 9: Get best layer ==="
BEST_LAYER1=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME3" --type "$TYPE1" --try_layers "$TRY_LAYERS")
echo "Best layer (cot): $BEST_LAYER1"

BEST_LAYER2=$(uv run -m utils.get_best_layer --model_name "$MODEL_NAME3" --type "$TYPE2" --try_layers "$TRY_LAYERS")
echo "Best layer (baseline): $BEST_LAYER2"

echo "=== Step 10: Final batch generation ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME3" --input_csv "test_harmful_prompts.csv" --type "$TYPE1" --layer "$BEST_LAYER1"

uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME3" --input_csv "test_harmful_prompts.csv" --type "$TYPE2" --layer "$BEST_LAYER2"

echo "=== Step 11: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME3" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE1}_layer_${BEST_LAYER1}.csv"

uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME3" --input_dir attack_results --input_csv "ortho_output_test_harmful_prompts_${TYPE2}_layer_${BEST_LAYER2}.csv"

echo "=== Step 12: Plot boxplot comparison ==="
uv run -m utils.plot_boxplot_comparison --model_name "$MODEL_NAME3" --type "all" --layer "$BEST_LAYER1,$BEST_LAYER2"
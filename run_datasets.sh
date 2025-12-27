#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


# Create log file with timestamp
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/experiment_${TIMESTAMP}.log"

# Log to both file and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Model: $MODEL_NAME"
echo ""

echo "=== Step 0: Create dataset ==="
uv run -m utils.create_base_dataset --dataset orbench --n 500 --dataset_dir dataset/base/
uv run -m utils.create_base_dataset --dataset strongreject --dataset_dir dataset/base/
uv run -m utils.create_base_dataset --dataset harmbench --dataset_dir dataset/base/
uv run -m utils.create_base_dataset --dataset advbench --dataset_dir dataset/base/
uv run -m utils.create_base_dataset --dataset sorrybench --dataset_dir dataset/base/

uv run  -m utils.check_duplicates

uv run -m utils.create_holdout_set --train_set_split 0.75 --input_csv all_harmful_prompts.csv --seed 42

uv run -m utils.create_random_subset --num_prompts 5 --seed 42

echo "=== Step 1: Generate clean model datasets ==="
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "train_harmful_prompts.csv" 
uv run -m utils.batch_generation_cot_output --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts.csv" 

echo "=== Step 2: Compute score outputs ==="
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_csv "train_harmful_prompts_cot5_out5.csv" -input_dir dataset
uv run -m utils.compute_score_outputs --model_name "$MODEL_NAME" --input_csv "test_harmful_prompts_cot5_out5.csv" -input_dir dataset

echo "=== Step 3: Create refusal and non-refusal datasets (both cot and baseline) ==="
uv run -m utils.filter_all_datasets --model_name "$MODEL_NAME" --scored_csv "scored_train_harmful_prompts_cot5_out5.csv"


#!/usr/bin/env bash
set -euo pipefail

HF_USERNAME=

MODEL_NAMES=(
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    Qwen/Qwen3-8B
    openai/gpt-oss-20b
)

DATASETS=(
    train_harmful_prompts_cot5_out5
    scored_train_harmful_prompts_cot5_out5
    test_harmful_prompts_cot5_out5
    scored_test_harmful_prompts_cot5_out5
    orbench_extra_prompts_cot5_out5
    scored_orbench_extra_prompts_cot5_out5
)

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"

    for j in "${!DATASETS[@]}"; do
        DATASET="${DATASETS[$j]}"
        uv run -m utils.csv_from_hf --hf_username "$HF_USERNAME" --model_name "$MODEL_NAME" --dataset "$DATASET"
    done
done

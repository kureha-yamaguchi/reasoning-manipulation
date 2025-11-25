#!/bin/bash

# Script to run multiple Python commands sequentially overnight
# Make it executable with: chmod +x run_overnight.sh

echo "Starting overnight batch processing at $(date)"
echo "================================================"

# First command
echo "Running batch generation at $(date)"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m utils.batch_generation_cot_output \
    --input_dir dataset/ \
    --input_csv all_harmful_prompts.csv \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Check if first command succeeded
if [ $? -eq 0 ]; then
    echo "Batch generation completed successfully at $(date)"
else
    echo "Batch generation failed at $(date)"
    exit 1
fi

echo "------------------------------------------------"

# Second command
echo "Running filter datasets at $(date)"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m utils.filter_datasets \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Check if second command succeeded
if [ $? -eq 0 ]; then
    echo "Filter datasets completed successfully at $(date)"
else
    echo "Filter datasets failed at $(date)"
    exit 1
fi

echo "================================================"
echo "All tasks completed at $(date)"
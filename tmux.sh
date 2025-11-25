# Set environment variables for RunPod
export HF_HOME=./huggingface_cache
export TRANSFORMERS_CACHE=./huggingface_cache
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN = 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

# Run your exact command with tee for live output + logging
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m utils.batch_generation_cot_output \
  --input_dir dataset/ \
  --input_csv all_harmful_prompts.csv \
  --model_name mistralai/Magistral-Small-2506 \
  --tensor_parallel_size 4 \
  --batch_size 64 \
  --temperature 0.7 \
  --gpu_memory_utilization 0.9 \
  2>&1 | tee generation.log

# # Run your exact command with tee for live output + logging
# CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python -m utils.batch_generation_cot_output \
#   --input_dir dataset/ \
#   --input_csv all_harmful_prompts.csv \
#   --model_name nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
#   --tensor_parallel_size 4 \
#   --batch_size 64 \
#   --temperature 0.6 \
#   --gpu_memory_utilization 0.9 \
#   2>&1 | tee generation.log


# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     python -m utils.filter_datasets --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#     2>&1 | tee filter.log


# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python -m utils.cache_activations \
#   --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#   --layers 14,15,16,17,18 \
#   --type cot \
#   2>&1 | tee activations.log
#!/bin/bash

# Configuration
WATCH_SESSION="generation"  # Name of session to watch
NEXT_COMMAND="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m utils.filter_datasets --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --batch_size 64"
CHECK_INTERVAL=1000  # Check every 60 seconds

echo "Watching session: $WATCH_SESSION"
echo "Will run after completion: $NEXT_COMMAND"

# Wait for the session to complete
while tmux has-session -t $WATCH_SESSION 2>/dev/null; do
    echo "$(date): Session $WATCH_SESSION still running..."
    
    # Optional: Show last line of output
    tmux capture-pane -t $WATCH_SESSION -p | tail -1
    
    sleep $CHECK_INTERVAL
done

echo "$(date): Session $WATCH_SESSION completed!"
echo "Starting next command..."

# Run the next command
export HF_HOME=./huggingface_cache
export TRANSFORMERS_CACHE=./huggingface_cache
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
eval $NEXT_COMMAND
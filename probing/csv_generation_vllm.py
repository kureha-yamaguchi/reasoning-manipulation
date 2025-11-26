"""
Generates model outputs from the locally stored orthogonalised model using vllm.

Usage (for multiple layers, subset holdout dataset):
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
uv run -m probing.csv_generation_vllm --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --type cot --layers 16,17,18,19 --eval_csv subset_5_test_harmful_prompts.csv

Usage (for chosen single layer, full holdout dataset):
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
uv run -m probing.csv_generation_vllm --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --type cot --layers 17 --eval_csv test_harmful_prompts.csv
"""


import argparse
import gc
from typing import List, Union
import os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through the model using vLLM - supports multiple CSV files"
    )
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name of the model (used to construct local path)")
    parser.add_argument("--type", type=str, required=True,
                        help="Ortho model from direction extracted from CoT tokens (cot) or 3 tokens at the end of prompt (baseline) or whole prompt (prompt)")
    parser.add_argument("--layers", type=str, default="17", 
                        help="Layers to take the activations and generate outputs from (comma-separated, e.g., '16,17,18,19')")
    parser.add_argument("--eval_csv", type=str, required=True,
                        help="CSV file to evaluate on")
    parser.add_argument("--max_new_tokens", type=int, default=2048, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, 
                        help="Top-p value for nucleus sampling")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, 
                        help="GPU memory utilization ratio")
    return parser.parse_args()

def read_csv(input_csv: str) -> Union[List[str], None]:
    """Read prompts from the CSV file."""
    print(f"Reading prompts from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        prompts = df['prompt'].tolist()
        print(f"Loaded {len(prompts)} prompts from the CSV file")
        return prompts
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return None

def save_csv(results: List[dict], output_csv: str):
    """Save results to CSV."""
    print(f"\nSaving results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        print(f"Results saved successfully to {output_csv}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

def apply_chat_template_batch(prompts: List[str], tokenizer) -> List[str]:
    """Apply chat template to a batch of prompts."""
    formatted_prompts = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            chat, 
            add_generation_prompt=True, 
            tokenize=False
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts

def process_batch(llm: LLM, tokenizer, prompts_batch: List[str], 
                 sampling_params: SamplingParams) -> List[dict]:
    """Process a batch of prompts."""
    results = []
    
    # Apply chat template to all prompts
    formatted_prompts = apply_chat_template_batch(prompts_batch, tokenizer)
    
    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    
    # Generate responses
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Process outputs
    for i, output in enumerate(outputs):
        original_prompt = prompts_batch[i]
        response = output.outputs[0].text
        
        results.append({
            "prompt": original_prompt,
            "response": response
        })
    
    return results

def process_csv(llm: LLM, tokenizer, input_csv: str, output_csv: str, args, sampling_params: SamplingParams) -> None:
    
    # Read prompts
    prompts = read_csv(input_csv)
    if prompts is None:
        print(f"Skipping {input_csv} due to read error.")
        return
    
    print(f"Processing {len(prompts)} prompts...")
    
    results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), args.batch_size):
        batch_end = min(i + args.batch_size, len(prompts))
        prompts_batch = prompts[i:batch_end]
        
        print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(prompts) + args.batch_size - 1)//args.batch_size}")
        print(f"Batch range: {i+1}-{batch_end} of {len(prompts)} prompts")
        
        batch_results = process_batch(llm, tokenizer, prompts_batch, sampling_params)
        results.extend(batch_results)
        
        # Clean up memory
        gc.collect()
    
    # Save results
    save_csv(results, output_csv)
    print(f"Completed processing {input_csv}! Generated {len(results)} total responses.")

    # Free memory
    del results, prompts, prompts_batch, batch_results, response, batch_end

    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    layers = [int(layer) for layer in args.layers.split(',')]

    for layer in layers:

        # Construct local model path
        local_model_path = os.path.join('results', args.model_name, f'ortho_model_{args.type}_layer_{layer}')
        print(f"Loading model from local path: {local_model_path}")
        
        # Verify the model directory exists
        if not os.path.exists(local_model_path):
            print(f"Error: Model directory does not exist: {local_model_path}")
            return
        
        # Handle CSV file selection
        
        input_csv = os.path.join('dataset', args.eval_csv)
        output_csv = os.path.join('results', args.model_name, 'attack_results', f'ortho_model_output_{args.type}_layer_{layer}.csv')

        # Initialize model and tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        
        print("Initializing vLLM...")
        llm = LLM(
            model=local_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            # max_model_len=4096,  # Adjust based on your model's context length
        )
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            # top_p=args.top_p
        )
        
        process_csv(llm, tokenizer, input_csv, output_csv, args, sampling_params)
        
        # Cleanup vLLM instance and GPU resources before next layer
        print(f"\nCleaning up resources for layer {layer}...")
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        # Synchronize CUDA operations to ensure cleanup is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        print(f"Cleanup complete for layer {layer}. Moving to next layer...\n")



def run():
    main()

if __name__ == "__main__":
    main()
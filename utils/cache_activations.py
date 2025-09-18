"""
Cache residual stream activations from a number of specified layers
Depending on the argument specified in --type, the following is cached:
if 'cot': average activation is taken across all cot token activations up to and including </think>
if 'baseline': average activation is taken across 3 tokens at the end of prompt
if 'prompt': average activation is taken across all prompt token activation up to and including <think>

Usage:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m utils.cache_activations \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --layers 14,15,16,17,18 \
    --type baseline
    --refusal_dataset train_refusal_0.05_baseline.csv
    --nonrefusal_dataset train_nonrefusal_0.6_baseline.csv
"""
import argparse
import gc
import os
import numpy as np
import pandas as pd
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract residual stream activations from language models")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B', 
                        help='Model name')
    parser.add_argument('--layers', type=str, default='15,19,23,27,31',
                        help='Comma-separated list of layer numbers to extract activations from')
    parser.add_argument('--type', type=str, default='baseline', 
                        help="CoT tokens (cot) or 3 tokens at the end of prompt (baseline) or whole prompt (prompt)")
    parser.add_argument('--refusal_dataset', type=str, default='train_refusal_0.05_pct0.75.csv', 
                        help="Name of refusal dataset")
    parser.add_argument('--nonrefusal_dataset', type=str, default='train_nonrefusal_0.6_pct0.75.csv', 
                        help="Name of non-refusal dataset")
    return parser.parse_args()

def cache_activations(model_name, dataset, layers, type, tokenizer, refusal=None):
    """
    Extract and cache residual stream activations from specified layers of a language model.
    
    This function processes a dataset of prompts and responses, extracts neural network
    activations from specified transformer layers, and saves them as numpy arrays.
    
    Args:
        model_name (str): HuggingFace model identifier (e.g., 'Qwen/Qwen3-8B')
        dataset (str): Dataset name ('refusal' or 'non_refusal') - determines input file path
        layers (list of int): List of layer indices to extract activations from
        type (str): Extraction mode:
            - 'cot': Average activations across Chain-of-Thought response tokens
            - 'baseline': Average activations across last 3 prompt tokens  
            - 'prompt': Average activations across all prompt tokens
        tokenizer: The tokenizer instance to use for encoding text
    """
    if refusal:
        # Create output directory if it doesn't exist
        output_dir = os.path.join('results', model_name, 'activations', 'refusal')
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Create output directory if it doesn't exist
        output_dir = os.path.join('results', model_name, 'activations', 'non_refusal')
        os.makedirs(output_dir, exist_ok=True)

    input_path = os.path.join('results', model_name, 'dataset', dataset)
    df = pd.read_csv(input_path)

    print(f"Processing {len(df)} examples from {dataset}")

    # Initialize model
    print(f"Initializing model {model_name}")
    model = LanguageModel(model_name, device_map="auto")

    # Initialize dictionary to store activation matrices for each layer
    activation_matrices = {layer: [] for layer in layers}
    print(f"Caching activations mode: {type}")

    # Process each example
    for idx, row in enumerate(tqdm(df.itertuples())):
        chat = [{"role": "user", "content": row.prompt}]
        # prompt_tokens = model.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        prompt_tokens = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        
        if type == 'cot':
            # Encode the cot response separately
            # response_tokens = model.tokenizer.encode(row.cot, add_special_tokens=False)
            response_tokens = tokenizer.encode(row.cot, add_special_tokens=False)
            # We want all tokens of the CoT (response)
            tokens_to_process = prompt_tokens + response_tokens
            target_start = len(prompt_tokens)  # Start of CoT
            target_end = len(tokens_to_process)  # End of our selection
        elif type == 'baseline':
            tokens_to_process = prompt_tokens
            target_start = max(0, len(prompt_tokens) - 3)  # Last 3 tokens of prompt
            target_end = len(prompt_tokens)
        elif type == 'prompt':
            tokens_to_process = prompt_tokens
            target_start = 0
            target_end = len(prompt_tokens)
        else:
            print("WARNING args.type not selected. Your choices are cot, baseline, prompt.")

        # Process the entire sequence at once
        # input_text = model.tokenizer.decode(tokens_to_process)
        input_text = tokenizer.decode(tokens_to_process)
        
        # Initialize dict to collect activations for this example across all layers
        example_layer_activations = {layer: [] for layer in layers}
        
        with torch.no_grad():
            with model.trace(input_text):
                for layer in layers:
                    # Note: Different models may have different attribute names
                    # For Qwen3, you might need to adjust this path
                    activation = model.model.layers[layer].input_layernorm.input.save()
                    example_layer_activations[layer].append(activation)
        
        # Compute means and add to matrices
        for layer in layers:
            layer_activations = example_layer_activations[layer][0]
            select_tokens = layer_activations[:, target_start:target_end, :]
            
            # print(f"DEBUGGING: Selected tokens shape: {select_tokens.shape}")
            
            # Compute mean across tokens (dimension 1)
            # mean_activation = torch.mean(select_tokens, dim=1).detach().cpu().numpy()
            mean_activation = torch.mean(select_tokens, dim=1).detach().cpu().to(torch.float32).numpy() # float32 for bfloat16 compatability
            activation_matrices[layer].append(mean_activation.squeeze())
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Save activation matrices for each layer
    for layer, activations in activation_matrices.items():
        if activations:
            activation_matrix = np.stack(activations)
            output_path = os.path.join(output_dir, f"layer_{layer}_{type}_activations.npy")
            np.save(output_path, activation_matrix)
            
            print(f"Saved activation matrix for layer {layer} with shape {activation_matrix.shape} to {output_path}")
    
    print("Extraction complete.")
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    # Parse layers
    layers = [int(layer) for layer in args.layers.split(',')]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('results', args.model_name, 'activations')
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer separately
    print(f"Loading tokenizer for {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Process both datasets
    cache_activations(model_name=args.model_name, dataset=args.refusal_dataset, layers=layers, type=args.type, tokenizer=tokenizer, refusal=True)
    cache_activations(model_name=args.model_name, dataset=args.nonrefusal_dataset, layers=layers, type=args.type, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
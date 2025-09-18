"""
This script visualizes the cosine similarity between a pre-computed direction vector
and per-token activations from layer 18 of DeepSeek-R1-Distill-Llama-8B.
"""
# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m probing.analyse_vectors --index 1 --type cot --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B

import argparse
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nnsight import LanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize activation similarities")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Name of the model you are interested in")
    parser.add_argument("--type", type=str, default="cot", help="Which activations were taken to compute the difference-of-means direction?")
    parser.add_argument('--dataset', type=str, default='heatmap_test_refusal_cot.csv',
                        help='Path to the dataset')
    parser.add_argument("--layer", type=int, default=17, 
                        help="Layer to extract activations from")
    parser.add_argument("--index", type=int, default=3, 
                        help="Index from cautious.csv")
    
    return parser.parse_args()

def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

def get_activations(model, row, layer=18):
    """Extract activations from specified layer using NNsight."""
    # Clear memory before processing
    gc.collect()
    torch.cuda.empty_cache()
    chat = [{"role": "user", "content": row.prompt}]
    prompt_tokens = model.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    cot_tokens = model.tokenizer.encode(row.cot, add_special_tokens=False)
    output_tokens = model.tokenizer.encode(row.output, add_special_tokens=False)
    tokens_to_process = prompt_tokens + cot_tokens + output_tokens
    input_text = model.tokenizer.decode(tokens_to_process)
    
    # Tokenize input text
    token_texts = [model.tokenizer.decode([token]) for token in tokens_to_process]

    # Trace the model to get activations
    with torch.no_grad():
            with model.trace(input_text):
                activation = model.model.layers[layer].input_layernorm.input.save()
    
    print(f"Activation tensor shape: {activation.shape}")
    
    # If tensor is flattened, try to reshape it
    if len(activation.shape) == 1:
        print("len(activation_tensor.shape) == 1")
        # Calculate the expected sequence length
        seq_len = len(input_text)
        hidden_size = 4096  # Expected hidden size for DeepSeek-R1-Distill-Llama-8B # TODO: make this configurable / automatic
        
        print("Tensor appears to be flattened. Attempting reshape...")
        print(f"Expected shape: [{seq_len}, {hidden_size}]")
        
        # Check if reshaping is possible
        if activation.numel() == seq_len * hidden_size:
            activation = activation.reshape(seq_len, hidden_size)
            print(f"Reshaped tensor to: {activation.shape}")
        else:
            print("WARNING: Cannot reshape tensor to expected dimensions.")
            print(f"Tensor has {activation.numel()} elements, but expected {seq_len * hidden_size}")
    
    # Clear cache after processing
    gc.collect()
    torch.cuda.empty_cache()
    
    return activation, token_texts

def compute_cosine_similarities(activation_tensor, direction_vector):
    """Compute cosine similarity between direction vector and each token's activation."""
    # Clear memory before computation
    gc.collect()
    torch.cuda.empty_cache()
    
    # Remove batch dimension if present
    if len(activation_tensor.shape) == 3:
        activation_tensor = activation_tensor.squeeze(0)  # Now shape is [463, 4096]
    
    # Ensure direction vector has the right shape
    direction_vector = direction_vector.reshape(1, -1)  # Shape [1, 4096]
    
    # Compute cosine similarity for each token
    similarities = []
    for token_idx in range(activation_tensor.shape[0]):
        token_activation = activation_tensor[token_idx].reshape(1, -1)  # Shape [1, 4096]
        similarity = torch.nn.functional.cosine_similarity(
            token_activation, direction_vector, dim=1
        ).item()
        similarities.append(similarity)
    
    # Clear memory after computation
    gc.collect()
    torch.cuda.empty_cache()
    
    return similarities

def plot_heatmap(token_texts, similarities, title, output_path="similarity_heatmap.png"):
    """Create a heatmap visualization of token similarities."""
    plt.figure(figsize=(12, 4))
    
    # Reshape similarities for heatmap (as a row)
    sim_matrix = np.array(similarities).reshape(1, -1)
    
    # Create custom x-tick labels - only show specific special tokens and every 10th token
    custom_positions = []
    
    for i, token in enumerate(token_texts):
        # Check if it's one of the specific special tokens or every 10th token
        special_tokens = ['<｜User｜>', '<｜Assistant｜>', '</think>']
        if token in special_tokens or ((i+1) % 10 == 0):
            custom_positions.append(i)
    
    # Create heatmap without x-tick labels initially
    ax = sns.heatmap(sim_matrix, cmap='coolwarm_r', center=0, 
                   xticklabels=False, yticklabels=["Similarity"], 
                   vmin=-0.4, vmax=0.4)
    
    # Set custom tick positions and labels
    ax.set_xticks(custom_positions)
    ax.set_xticklabels([token_texts[i] for i in custom_positions], rotation=90, fontsize=8)

    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() in special_tokens:
            tick_label.set_color('blue')
            tick_label.set_weight('bold')
    
    plt.title(f'Prompt: {title}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    plt.close()
    
    # Clear memory after plotting
    gc.collect()

def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()
    model_name = args.model_name

    # Set plotting settings
    set_plotting_settings()
    
    # Load the pre-computed direction vector
    direction_vector = torch.load(os.path.join('results', model_name, 'refusal_dir', f'{args.type}_refusal_dir.pt'))
    print(f"Loaded direction vector with shape: {direction_vector.shape}")
    
    dataset_path = os.path.join('results', model_name, 'dataset', args.dataset)
    print(f"Loading CSV dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    row = df.iloc[args.index]
    prompt = row.prompt
    print(prompt)

    model = LanguageModel(model_name, device_map="auto")
    # Memory cleanup after model loading
    gc.collect()
    torch.cuda.empty_cache()

    # Get activations for the input text
    print(f"Extracting activations from layer {args.layer}")
    activations, token_texts = get_activations(model, row, layer=args.layer)
    
    # Compute cosine similarities
    similarities = compute_cosine_similarities(activations, direction_vector)
    
    # Create visualizations
    # plot_token_similarities(token_texts, similarities, 
    #                        output_path=os.path.join(args.output_dir, "after_4_bars.png"))
    output_dir = os.path.join('results', model_name, 'figures')
    plot_heatmap(token_texts, similarities, prompt,
                output_path=os.path.join(output_dir, f"{args.dataset}_{args.index}.png"))
    
    # Print highest and lowest similarity tokens
    sorted_indices = np.argsort(similarities)
    print("\nTokens with highest similarity:")
    for idx in sorted_indices[-5:]:
        print(f"  {token_texts[idx]}: {similarities[idx]:.4f}")
    
    print("\nTokens with lowest similarity:")
    for idx in sorted_indices[:5]:
        print(f"  {token_texts[idx]}: {similarities[idx]:.4f}")
    
    # Final memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

def run():
    main()

if __name__ == "__main__":
    main()
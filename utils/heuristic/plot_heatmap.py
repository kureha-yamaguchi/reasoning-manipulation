'''
Plot average strongreject score as a function of sentence_idx.

Example usage:
uv run -m utils.heuristic.plot_heatmap \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --index_number 3 \
    --cot_number 1
'''
import csv
import os
import argparse
from typing import List, Dict, Tuple, Any

from tqdm import tqdm
from collections import defaultdict
import statistics 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import gc
import re

import numpy as np
import seaborn as sns
import torch
from nnsight import LanguageModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average strongreject score vs sentence index"
    )
    parser.add_argument("--model_name", type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                        help="Path to scored CSV file")
    parser.add_argument("--index_number", type=int, required=True,
                        help="Index number as per quadrant_output.txt")
    parser.add_argument("--cot_number", type=int, required=True,
                        help="Cot number as per quadrant_output.txt (x/5)")
    parser.add_argument("--type", type=str, default='cot',
                        help="Determines which refusal direction to use")
    parser.add_argument("--layer", type=int, default=17,
                        help="Determines which refusal direction to use")
    parser.add_argument("--repetitions", type=int, default=15,
                        help="Number of output variations per prompt")
    return parser.parse_args()


def load_scored_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Load CSV data with pre-computed scores.
    
    Args:
        csv_path: Path to the CSV file containing scores
    
    Returns:
        List of dictionaries containing all CSV data including scores
    """
    print(f"Loading scored data from: {csv_path}")
    
    # First pass: count rows for progress bar
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for _ in reader)
    
    # Second pass: load data with progress bar
    all_rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_rows, desc="Loading scored data"):
            all_rows.append(row)
    
    print(f"Loaded {len(all_rows)} rows")
    return all_rows


def compute_stats_per_prompt_cot(
    scored_rows: List[Dict[str, str]]
) -> Tuple[List[List[float]], List[List[float]], List[List[str]]]:
    """
    Groups by (prompt, cot_rep_n), then nests results by prompt.
    
    Returns:
        Tuple of (means_per_prompt, std_devs_per_prompt, cots_per_prompt) where each is a 
        list of lists: [[values for prompt1's CoTs], [values for prompt2's CoTs], ...]
    """
    # Group by (prompt, cot_rep_n) to find all outputs for each CoT
    cot_groups = defaultdict(list)
    for row in tqdm(scored_rows, desc="Grouping by (prompt, cot_rep_n)"):
        key = (row['prompt'], row['cot_rep_n'])
        cot_groups[key].append(row)

    # Compute stats per (prompt, cot), organized by prompt
    prompt_to_stats = defaultdict(lambda: {'means': [], 'std_devs': [], 'cots': []})
    
    for (prompt, cot_rep_n), rows in tqdm(cot_groups.items(), desc="Processing CoT groups"):
        chunk_scores = [float(row['strongreject_score']) for row in rows]
        
        mean = statistics.mean(chunk_scores)
        std_dev = statistics.stdev(chunk_scores) if len(chunk_scores) > 1 else 0.0
        
        # Extract the unique CoT text (should be the same for all rows in this group)
        cot_text = rows[0]['cot']
        
        prompt_to_stats[prompt]['means'].append(mean)
        prompt_to_stats[prompt]['std_devs'].append(std_dev)
        prompt_to_stats[prompt]['cots'].append(cot_text)

    # Convert to lists of lists (preserving prompt order if needed)
    means = [stats['means'] for stats in prompt_to_stats.values()]
    std_devs = [stats['std_devs'] for stats in prompt_to_stats.values()]
    cots = [stats['cots'] for stats in prompt_to_stats.values()]
    
    return means, std_devs, cots


def compute_stats_per_prompt(
    scored_rows: List[Dict[str, str]]
) -> Tuple[List[float], List[float], List[str]]:
    """
    This function groups by prompt only — aggregating across all CoTs and outputs. 
    The standard deviation here (std_dev_i) captures total variance (from both CoT 
    variation and output sampling).
    
    Args:
        scored_rows: List of dictionaries containing CSV data with scores
    
    Returns:
        Tuple of (means, std_devs, prompts)
    """
    means = []
    std_devs = []
    prompts = []

    # Group by prompt to find all outputs for each prompt
    prompt_groups = defaultdict(list)
    for row in tqdm(scored_rows, desc="Grouping by prompt"):
        key = row['prompt']
        prompt_groups[key].append(row)

    # Process for each prompt
    for prompt, rows in tqdm(prompt_groups.items(), desc="Processing prompt groups"):
        # Extract scores for this prompt
        chunk_scores = [float(row['strongreject_score']) for row in rows]

        mean = statistics.mean(chunk_scores)
        std_dev = statistics.stdev(chunk_scores) if len(chunk_scores) > 1 else 0.0

        # Append to lists
        means.append(mean)
        std_devs.append(std_dev)
        prompts.append(prompt)
        
    return means, std_devs, prompts


def find_quadrant(scored_rows, x_threshold=0.03, y_threshold=0.35):
    quadrant_points = []
    means_ij, std_devs_ij, cots = compute_stats_per_prompt_cot(scored_rows)
    # Average per prompt first, then overall average
    per_prompt_avgs = [statistics.mean(sublist) for sublist in std_devs_ij]
    means_i, std_devs_i, prompts = compute_stats_per_prompt(scored_rows)

    n = len(per_prompt_avgs)

    for i in range(n):
        if per_prompt_avgs[i] < x_threshold and std_devs_i[i] > y_threshold:
            dict = {
                "prompt_idx": i,
                "prompt": prompts[i],
                "cots": cots[i],
                "average mean per CoT": [f"{x:.2f}" for x in means_ij[i]],
                "std dev conditioned on CoT": [f"{x:.2f}" for x in std_devs_ij[i]],
                "std dev conditioned on prompt": f"{std_devs_i[i]:.2f}"
            }
            quadrant_points.append(dict)
    
    return quadrant_points


def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": False,  # Disable autolayout for GridSpec
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)


def split_cot_into_sentences(cot: str) -> List[str]:
    """
    Split a chain-of-thought text into individual sentences.
    
    Args:
        cot: The chain-of-thought text string
        
    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, cot.strip())
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def get_activations(model, quadrant_points, index_number=19, cot_number=1, layer=18):
    """Extract activations from specified layer using NNsight."""
    # Clear memory before processing
    gc.collect()
    torch.cuda.empty_cache()

    index_number -= 1
    cot_number -= 1
    prompt = quadrant_points[index_number]['prompt']
    cot = quadrant_points[index_number]['cots'][cot_number]

    # Split CoT into sentences
    sentences = split_cot_into_sentences(cot)
    
    chat = [{"role": "user", "content": prompt}]
    prompt_tokens = model.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    
    # Tokenize each sentence separately and track boundaries
    cot_tokens = []
    sentence_token_boundaries = []  # List of (start_idx, end_idx) for each sentence in cot_tokens
    
    for sentence in sentences:
        sentence_tokens = model.tokenizer.encode(sentence, add_special_tokens=False)
        start_idx = len(cot_tokens)
        cot_tokens.extend(sentence_tokens)
        end_idx = len(cot_tokens)
        sentence_token_boundaries.append((start_idx, end_idx))
    
    tokens_to_process = prompt_tokens + cot_tokens
    input_ids = torch.tensor([tokens_to_process])
    
    # Tokenize input text
    token_texts = [model.tokenizer.decode([token]) for token in tokens_to_process]
    
    # Calculate the offset for CoT tokens (after prompt tokens)
    prompt_offset = len(prompt_tokens)

    # Trace the model to get activations
    with torch.no_grad():
            with model.trace(input_ids):
                activation = model.model.layers[layer].input_layernorm.input.save()
    
    print(f"Activation tensor shape: {activation.shape}")
    print(f"Number of sentences in CoT: {len(sentences)}")
    print(f"Sentence token boundaries: {sentence_token_boundaries}")
    
    # If tensor is flattened, try to reshape it
    if len(activation.shape) == 1:
        print("len(activation_tensor.shape) == 1")
        # Calculate the expected sequence length
        seq_len = len(input_ids)
        hidden_size = model.config.hidden_size
        
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
    
    return activation, token_texts, sentences, sentence_token_boundaries, prompt_offset


def compute_cosine_similarities(activation_tensor, direction_vector):
    """Compute cosine similarity between direction vector and each token's activation."""
    # Clear memory before computation
    gc.collect()
    torch.cuda.empty_cache()
    
    # Remove batch dimension if present
    if len(activation_tensor.shape) == 3:
        activation_tensor = activation_tensor.squeeze(0)
    
    # Ensure direction vector has the right shape
    direction_vector = direction_vector.reshape(1, -1)
    
    # Compute cosine similarity for each token
    similarities = []
    for token_idx in range(activation_tensor.shape[0]):
        token_activation = activation_tensor[token_idx].reshape(1, -1)
        similarity = torch.nn.functional.cosine_similarity(
            token_activation, direction_vector, dim=1
        ).item()
        similarities.append(similarity)
    
    # Clear memory after computation
    gc.collect()
    torch.cuda.empty_cache()
    
    return similarities


def compute_sentence_similarities(similarities: List[float], 
                                   sentence_token_boundaries: List[Tuple[int, int]], 
                                   prompt_offset: int) -> List[float]:
    """
    Compute average similarity for each sentence by averaging over its tokens.
    
    Args:
        similarities: List of similarities for each token
        sentence_token_boundaries: List of (start_idx, end_idx) for each sentence in CoT
        prompt_offset: Number of tokens in the prompt (before CoT starts)
        
    Returns:
        List of average similarities for each sentence
    """
    sentence_similarities = []
    
    for start_idx, end_idx in sentence_token_boundaries:
        # Adjust indices to account for prompt tokens
        global_start = prompt_offset + start_idx
        global_end = prompt_offset + end_idx
        
        # Get similarities for this sentence's tokens
        sentence_token_sims = similarities[global_start:global_end]
        
        if sentence_token_sims:
            avg_sim = np.mean(sentence_token_sims)
        else:
            avg_sim = 0.0
            
        sentence_similarities.append(avg_sim)
    
    return sentence_similarities


def plot_combined_figure(input_csv, sentence_similarities, sentences, quadrant_points, 
                         output_dir, args, index_number=19, cot_number=1,repetitions=15):
    """
    Create a single unified figure with:
    - Line plot of mean StrongReject score with std deviation bands
    - Background cells colored according to sentence_similarities for each sentence index
    
    The x-axis uses sentence_idx values directly from the CSV file.
    Each sentence index has a colored vertical band behind the line plot.
    
    Args:
        input_csv: Path to the CSV file with resampling results
        sentence_similarities: List of average similarities per sentence
        sentences: List of sentence texts
        quadrant_points: Data structure with prompt information
        output_dir: Directory to save the plot
        args: Command line arguments
        index_number: Index of the quadrant point (1-indexed)
        cot_number: CoT rollout number (1-indexed)
    """
    # Load data for the line plot
    input_path = os.path.join('results', args.model_name, 'dataset', input_csv)
    df = pd.read_csv(input_path)
    
    # Compute mean and std strongreject_score per sentence_idx
    mean_scores = df.groupby('sentence_idx')['strongreject_score'].mean()
    std_scores = df.groupby('sentence_idx')['strongreject_score'].std().fillna(0)  # Fill NaN with 0
    
    # Get the actual sentence_idx values from the CSV (this is the shared x-axis)
    sentence_indices = mean_scores.index.values
    
    # Ensure sentence_similarities aligns with sentence_indices from CSV
    n_from_csv = len(sentence_indices)
    n_from_similarities = len(sentence_similarities)
    
    if n_from_csv != n_from_similarities:
        print(f"Warning: Number of sentence indices from CSV ({n_from_csv}) does not match "
              f"number of sentence similarities ({n_from_similarities})")
        # Align by using the minimum length
        min_len = min(n_from_csv, n_from_similarities)
        sentence_similarities = sentence_similarities[:min_len]
        sentence_indices = sentence_indices[:min_len]
        mean_scores = mean_scores.iloc[:min_len]
        std_scores = std_scores.iloc[:min_len]
    
    # Get metadata for title
    idx_number = index_number - 1
    cot_num = cot_number - 1
    title = quadrant_points[idx_number]['prompt']
    score = quadrant_points[idx_number]['average mean per CoT'][cot_num]
    std_dev = quadrant_points[idx_number]['std dev conditioned on CoT'][cot_num]
    
    # Determine x-axis range based on actual sentence indices from CSV
    x_min = sentence_indices.min()
    x_max = sentence_indices.max()
    n_sentences = len(sentence_indices)
    
    # Create single figure
    fig_width = max(14, n_sentences * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Set up colormap for background cells
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(vmin=-0.4, vmax=0.4)
    
    # Set fixed y-axis limits for Mean StrongReject Score (0 to 1)
    y_min = 0
    y_max = 1
    
    # Draw colored background cells for each sentence index
    for i, (idx, sim) in enumerate(zip(sentence_indices, sentence_similarities)):
        # Calculate cell boundaries (centered on sentence_idx)
        left = idx - 0.5
        right = idx + 0.5
        
        # Get color from colormap based on similarity value
        color = cmap(norm(sim))
        
        # Draw rectangle spanning full y-axis height
        rect = plt.Rectangle((left, y_min), right - left, y_max - y_min,
                              facecolor=color, edgecolor='none', alpha=0.6, zorder=0)
        ax.add_patch(rect)
        
        # Add similarity value as text at bottom of each cell
        ax.text(idx, y_min + (y_max - y_min) * 0.02, f'{sim:.2f}', 
                ha='center', va='bottom', fontsize=7, color='black', zorder=3)
    
    # Plot the line with std deviation band on top of colored cells
    ax.fill_between(sentence_indices, 
                    mean_scores.values - std_scores.values, 
                    mean_scores.values + std_scores.values, 
                    alpha=0.4, label='±1 Std Dev', color='#377eb8', zorder=1)
    ax.plot(sentence_indices, mean_scores.values, marker='o', linewidth=2, 
            markersize=6, label='Mean StrongReject Score', color='#377eb8', zorder=2)
    
    # Configure axes
    ax.set_ylabel('Mean Output StrongReject Score After Resampling', fontsize=12)
    ax.set_xlabel('CoT Sentence Index', fontsize=12)
    ax.set_title(f'StrongReject Score & Similarity by Sentence Index\n'
                 f'Prompt: {title}\n'
                 f'Avg mean per CoT: {float(score):.2f}, Std dev: {float(std_dev):.2f}, '
                 f'CoT rollout: {cot_number}\nNumber of resampling repetitions: {repetitions}', fontsize=11)
    
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min, y_max)
    
    # Set x-ticks to match sentence_idx values
    ax.set_xticks(sentence_indices)
    ax.set_xticklabels([str(idx) for idx in sentence_indices], 
                       rotation=45, ha='right', fontsize=8)
    
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc='upper right', fontsize=9)
    
    # Add colorbar for similarity values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Cosine Similarity with Refusal Direction', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # Adjust layout and save
    plt.tight_layout()
    
    base_name = os.path.splitext(input_csv)[0]
    output_path = os.path.join(output_dir, 
                               f'combined_plot_index{index_number}_cot{cot_number}.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    
    plt.show()
    plt.close()
    
    gc.collect()

def plot_sentence_heatmap(sentence_similarities, sentences, quadrant_points, output_dir, 
                          index_number=19, cot_number=1):
    """
    Create a heatmap visualization with sentence indices on the x-axis.
    Each cell represents the average similarity for that sentence.
    
    Args:
        sentence_similarities: List of average similarities per sentence
        sentences: List of sentence texts
        quadrant_points: Data structure with prompt information
        output_dir: Directory to save the plot
        index_number: Index of the quadrant point (1-indexed)
        cot_number: CoT rollout number (1-indexed)
    """
    plt.figure(figsize=(max(12, len(sentences) * 0.3), 4))
    output_path = os.path.join(output_dir, f"sentence_similarity_heatmap_index{index_number}_cot{cot_number}.png")

    idx_number = index_number - 1
    cot_num = cot_number - 1
    title = quadrant_points[idx_number]['prompt']
    score = quadrant_points[idx_number]['average mean per CoT'][cot_num]
    std_dev = quadrant_points[idx_number]['std dev conditioned on CoT'][cot_num]
    
    # Reshape similarities for heatmap (as a row)
    sim_matrix = np.array(sentence_similarities).reshape(1, -1)
    
    # Create sentence index labels
    sentence_labels = [f"S{i+1}" for i in range(len(sentences))]
    
    # Create heatmap
    ax = sns.heatmap(sim_matrix, cmap='coolwarm_r', center=0, 
                     xticklabels=sentence_labels, yticklabels=["Avg Similarity"], 
                     vmin=-0.4, vmax=0.4, annot=True, fmt='.2f', annot_kws={'fontsize': 6})
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title(f'Sentence-level Similarity\nPrompt: {title[:60]}...\nAvg mean per CoT: {float(score):.2f}, Std dev: {float(std_dev):.2f}, CoT rollout: {cot_number}', fontsize=10)
    plt.xlabel('Sentence Index')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Sentence heatmap saved to {output_path}")
    plt.show()
    plt.close()
    
    # Print sentence details
    print("\nSentence breakdown:")
    for i, (sentence, sim) in enumerate(zip(sentences, sentence_similarities)):
        print(f"  S{i+1} (sim={sim:.3f}): {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
    
    gc.collect()

def plot_graph(model_name, input_csv):
    input_path = os.path.join('results', model_name, 'dataset', input_csv)
    base_name = os.path.splitext(input_csv)[0]
    output_path = os.path.join('results', model_name, 'figures', f'graph_{base_name}')
    # Load data
    df = pd.read_csv(input_path)
    
    # Compute mean and std strongreject_score per sentence_idx
    mean_scores = df.groupby('sentence_idx')['strongreject_score'].mean()
    std_scores = df.groupby('sentence_idx')['strongreject_score'].std()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_scores.index, mean_scores.values, marker='o', linewidth=2, markersize=6, label='Mean')
    plt.fill_between(mean_scores.index, 
                     mean_scores.values - std_scores.values, 
                     mean_scores.values + std_scores.values, 
                     alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Sentence Index', fontsize=12)
    plt.ylabel('Mean Output StrongReject Score After Resampling', fontsize=12)
    plt.title('Mean StrongReject Score After Resampling by Sentence Index', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

def main():
    args = parse_args()
    # Set plotting settings
    set_plotting_settings()

    # These variables need to be defined or passed appropriately
    results_dir = 'results'
    model_name = args.model_name
    index_number = args.index_number
    cot_number = args.cot_number
    layer = args.layer
    output_dir = os.path.join(results_dir, model_name, 'figures')

    scored_csv = "scored_train_harmful_prompts_cot5_out5.csv"
    scored_csv_path = os.path.join(results_dir, model_name, "dataset", scored_csv)

    scored_rows = load_scored_csv(scored_csv_path)
    quadrant_points = find_quadrant(scored_rows)

    # Load the pre-computed direction vector
    direction_vector = torch.load(os.path.join(results_dir, model_name, 'refusal_dir', f'refusal_dir_{args.type}_layer_{args.layer}.pt'))
    print(f"Loaded direction vector with shape: {direction_vector.shape}")

    model = LanguageModel(model_name, device_map="auto")

    # Get activations with sentence information
    # Note: get_activations function needs to be defined or imported
    activations, token_texts, sentences, sentence_token_boundaries, prompt_offset = get_activations(
        model, quadrant_points, index_number=index_number, cot_number=cot_number, layer=layer
    )

    # Compute cosine similarities for each token
    similarities = compute_cosine_similarities(activations, direction_vector)

    # Compute average similarity per sentence
    sentence_similarities = compute_sentence_similarities(
        similarities, sentence_token_boundaries, prompt_offset
    )

    # Use the new combined plotting function
    input_csv = f'scored_resampling_results_idx{args.index_number}_cot{args.cot_number}.csv'
    
    plot_combined_figure(
        input_csv=input_csv,
        sentence_similarities=sentence_similarities,
        sentences=sentences,
        quadrant_points=quadrant_points,
        output_dir=output_dir,
        args=args,
        index_number=index_number,
        cot_number=cot_number,
        repetitions=args.repetitions
    )


if __name__ == "__main__":
    main()
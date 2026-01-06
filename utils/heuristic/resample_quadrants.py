'''
Docstring for utils.heuristic.resampling.

1. Reads in rows from scored_train_harmful_prompts_cot5_out5.csv
2. Identifies points that lie in the quadrant. These are generations that have low standard deviation conditioned on the specific prompt-CoT but high standard deviation when conditioned only on the prompt.
3. Performs resampling with n rollouts for a given index_number and cot_number in the quadrant, beginning at cot sentence start_idx and ending at end_idx
4. Partitions into output (after close think tag)
5. Saves generations

So with start_idx=2, end_idx=4, n=5:

Generate 5 rollouts from: prompt + S1 + S2
Generate 5 rollouts from: prompt + S1 + S2 + S3
Generate 5 rollouts from: prompt + S1 + S2 + S3 + S4

Example usage:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.heuristic.resample_quadrants \
  --index_number 3\
  --cot_number 1 \
  --start_idx 10 \
  --end_idx 12
'''

import csv
import os
import argparse
from re import X
from typing import List, Dict, Tuple, Any

from tqdm import tqdm
from collections import defaultdict
import statistics 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import argparse
import gc
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple output rollouts per prompt for non-reasoning models"
    )
    parser.add_argument("--index_number", type=int, required=True,
                        help="Index number as per quadrant_output.txt")
    parser.add_argument("--cot_number", type=int, required=True,
                        help="Cot number as per quadrant_output.txt (x/5)")
    parser.add_argument("--start_idx", type=int, required=True,
                        help="Sentence number to start resampling from")
    parser.add_argument("--end_idx", type=int, required=True,
                        help="Sentence number to end resampling on")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model to use for generation")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--scored_csv", type=str, default='scored_train_harmful_prompts_cot5_out5.csv',
                        help="Scored CSV file with prompts")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of output variations per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio")
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
    # The lambda function is called: lambda: {'means': [], 'std_devs': [], 'cots': []}
    # It returns a fresh dictionary: {'means': [], 'std_devs': [], 'cots': []}
    # That dictionary is stored at prompt_to_stats['some_prompt']
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
    This function groups by prompt only — aggregating across all CoTs and outputs. The standard deviation here (std_dev_i) captures total variance (from both CoT variation and output sampling).
    
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


def split_cot_into_sentences(cot: str) -> List[str]:
    """
    Split a chain-of-thought text into individual sentences.
    
    Args:
        cot: The chain-of-thought text string
        
    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation followed by whitespace
    # This regex handles . ! ? followed by whitespace or end of string
    # It also handles cases like "..." and keeps the delimiter with the sentence
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, cot.strip())
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def get_sentences(quadrant_points, index_number=19, cot_number=1):
    """Extract activations from specified layer using NNsight."""
    # Clear memory before processing
    gc.collect()
    torch.cuda.empty_cache()

    index_number -= 1
    cot_number -= 1
    cot = quadrant_points[index_number]['cots'][cot_number]

    # Split CoT into sentences
    sentences = split_cot_into_sentences(cot)
    
    return sentences


def main():
    args = parse_args()

    if not args.tensor_parallel_size:
        args.tensor_parallel_size = torch.cuda.device_count()

    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    scored_csv_path = os.path.join(args.results_dir, args.model_name, "dataset", args.scored_csv)

    scored_rows_deepseek_qwen = load_scored_csv(scored_csv_path)
    quadrant_points = find_quadrant(scored_rows_deepseek_qwen)

    # Get activations with sentence information
    sentences= get_sentences(quadrant_points, index_number=args.index_number, cot_number=args.cot_number)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

    # Get the original prompt for reference
    original_prompt = quadrant_points[args.index_number-1]['prompt']
    chat = [{"role": "user", "content": original_prompt}]
    formatted_prompt= tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    print(repr(formatted_prompt))

    # initial cot. if start_idx=3, then includes S1 + S2
    initial_cot = ' '.join(sentences[:args.start_idx-1])
    combined_thread = formatted_prompt + initial_cot

    # Store results for CSV
    results = []
    
    # Get the original prompt for reference
    original_prompt = quadrant_points[args.index_number-1]['prompt']

    for i in range(args.end_idx - args.start_idx + 1):
        # if start_idx=3, this step now includes S3
        add_to_cot = sentences[args.start_idx-1+i]
        # augment the initial_cot
        combined_thread = ' '.join([combined_thread, add_to_cot])
        
        # Current sentence index (1-indexed for clarity)
        current_sentence_idx = args.start_idx + i
        
        print(f"\n{'='*60}")
        print(f"Resampling from sentence S{current_sentence_idx}")
        print(f"Generating {args.repetitions} rollouts...")
        
        # Generate all repetitions in one batched call
        outputs = llm.generate([combined_thread] * args.repetitions, sampling_params)
        
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            
            # Extract output after </think> tag
            think_pattern = r'(.*?</think>)(.*)'
            match = re.match(think_pattern, generated_text, re.DOTALL)
            
            # Only store result if </think> tag was found
            if match:
                output_text = match.group(2)
                results.append({
                    'prompt': original_prompt,
                    'sentence_idx': current_sentence_idx,
                    'resample_n': j + 1,
                    'output': output_text
                })
        
        print(f"  {args.repetitions} rollouts completed")

    # Save results to CSV
    output_csv_path = os.path.join(
        args.results_dir, 
        args.model_name, 
        "dataset",
        f"resampling_results_idx{args.index_number}_cot{args.cot_number}_s{args.start_idx}_e{args.end_idx}.csv"
    )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_csv_path}")
    print(f"Total rollouts generated: {len(results)}")


if __name__ == "__main__":
    main()
'''
utils.heuristic.resample_quadrants

Targeted resampling of "quadrant" prompts — those where the chain-of-thought
is the key determinant of harmful compliance.

Pipeline:
1. Loads pre-scored CSV files (prompt, cot, strongreject_score per row).
2. Computes two variance measures per prompt:
     - Within-CoT variance: std dev of output scores given a fixed (prompt, CoT).
     - Across-CoT variance: std dev of output scores across all CoTs for a prompt.
3. Selects "quadrant" prompts: low within-CoT variance (< x_threshold) AND
   high across-CoT variance (> y_threshold). These are prompts where each
   individual CoT reliably produces consistent outputs, but different CoTs
   lead to very different outcomes — i.e. the reasoning path is the switch.
4. Randomly samples n prompts from the quadrant and saves their indices.
5. For each sampled prompt and each of its CoTs, performs progressive
   resampling: iterates from 0 sentences of the CoT prefix up to the full CoT, generating `repetitions` rollouts at each position. Skips any prompt that already has output files on disk.
6. Saves rollouts to CSV files under <results_dir>/<model_name>/dataset/quadrants/.


Example usage:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.heuristic.resample_quadrants \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
'''

import csv
import os
import argparse
from typing import List, Dict, Tuple, Union

from tqdm import tqdm
from collections import defaultdict
import statistics 
import pandas as pd
import gc
import random
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple output rollouts per prompt for non-reasoning models"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model to use for generation")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--repetitions", type=int, default=10,
                        help="Number of output variations per prompt")
    parser.add_argument("--x_threshold", type=float, default=0.03,
                        help="Upper bound on x axis for selecting high-variance points")
    parser.add_argument("--y_threshold", type=float, default=0.43,
                        help="Lower bound on y axis for selecting high-variance points")
    parser.add_argument("--n", type=int, default=32,
                        help="Number of rows to randomly select from high-variance set")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio")
    parser.add_argument("--max_retries_multiplier", type=int, default=10,
                        help="Maximum total generations = repetitions * this multiplier (safety limit)")
    return parser.parse_args()


def load_scored_csv(csv_paths: Union[str, List[str]]) -> List[Dict[str, str]]:
    """
    Load CSV data with pre-computed scores from one or more files.
    
    Args:
        csv_paths: Path to a CSV file or list of paths to CSV files containing scores
    
    Returns:
        List of dictionaries containing all CSV data including scores
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    
    all_rows = []
    skipped_count = 0
    
    for csv_path in csv_paths:
        print(f"Loading scored data from: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            total_rows = sum(1 for _ in reader)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=total_rows, desc=f"Loading {os.path.basename(csv_path)}"):
                # Skip rows with empty or invalid scores
                if row.get('strongreject_score', '').strip():
                    all_rows.append(row)
                else:
                    skipped_count += 1
        
        print(f"Loaded rows from {os.path.basename(csv_path)}")
    
    print(f"Total loaded: {len(all_rows)} rows from {len(csv_paths)} file(s)")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} rows with empty scores")
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
    Groups by prompt only — aggregating across all CoTs and outputs.
    
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
        chunk_scores = [float(row['strongreject_score']) for row in rows]

        mean = statistics.mean(chunk_scores)
        std_dev = statistics.stdev(chunk_scores) if len(chunk_scores) > 1 else 0.0

        means.append(mean)
        std_devs.append(std_dev)
        prompts.append(prompt)
        
    return means, std_devs, prompts


def find_quadrant(scored_rows, x_threshold=0.03, y_threshold=0.43):
    """Find data points in the target quadrant based on variance thresholds."""
    quadrant_points = []
    means_ij, std_devs_ij, cots = compute_stats_per_prompt_cot(scored_rows)
    per_prompt_avgs = [statistics.mean(sublist) for sublist in std_devs_ij]
    means_i, std_devs_i, prompts = compute_stats_per_prompt(scored_rows)

    n = len(per_prompt_avgs)

    for i in range(n):
        if per_prompt_avgs[i] < x_threshold and std_devs_i[i] > y_threshold:
            point = {
                "prompt_idx": i,
                "prompt": prompts[i],
                "cots": cots[i],
                "average mean per CoT": [f"{x:.2f}" for x in means_ij[i]],
                "std dev conditioned on CoT": [f"{x:.2f}" for x in std_devs_ij[i]],
                "std dev conditioned on prompt": f"{std_devs_i[i]:.2f}"
            }
            quadrant_points.append(point)
    
    return quadrant_points


def split_cot_into_sentences(cot: str) -> List[str]:
    """
    PRESERVES LEADING WHITESPACES
    Split chain-of-thought text into sentences with leading whitespace.
    
    Each sentence after the first carries a leading space so that
    "".join(sentences) reconstructs the original spacing and tokens
    align with BPE's leading-whitespace convention.
    
    Args:
        cot: The chain-of-thought text string
        
    Returns:
        List of sentences. The first has no leading whitespace; all
        subsequent sentences have a single leading space.
    """
    # \s*          - leading whitespace (none for first sentence, space for rest)
    # .*?          - content (non-greedy)
    # [.!?]["']?   - sentence-ending punctuation + optional closing quote
    # (?=\s|$)     - lookahead: whitespace or end of string
    
    pattern = r'\s*.*?(?:[.!?]["\']?(?=\s|$)|</?think>)'
    sentences = re.findall(pattern, cot)
    
    return [s for s in sentences if s]


def get_sentences(quadrant_points, prompt_index):
    """Extract sentences from specified quadrant point."""
    sentences = []
    cots = next((item['cots'] for item in quadrant_points if item["prompt_idx"] == prompt_index), None)
    for cot in cots:
        sentences.append(split_cot_into_sentences(cot))
    
    return sentences


def extract_valid_output(generated_text: str, model_name: str = None) -> str | None:
    """
    Extract the output after the thinking/analysis section.
    
    Args:
        generated_text: The full generated text from the model
        model_name: Model name to determine the delimiter pattern
        
    Returns:
        The text after the thinking delimiter, or None if delimiter not found
    """
    if model_name == "openai/gpt-oss-20b":
        # GPT-OSS uses channel-based format
        delimiter = r'<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>'
        pattern = rf'(.*?{delimiter})(.*)'
    else:
        # DeepSeek and similar models use </think> tag
        pattern = r'(.*?</think>)(.*)'
    
    match = re.match(pattern, generated_text, re.DOTALL)
    
    if match:
        return match.group(2)
    return None


def generate_valid_outputs(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    sampling_params: SamplingParams,
    num_required: int,
    max_total_generations: int,
    is_last_sentence: bool,
    model_name: str = None
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Generate exactly num_required valid outputs.
    
    With high success rates, this typically completes in one batch.
    If some outputs are invalid (missing </think> tag), subsequent 
    iterations generate exactly the number still needed.
    
    Args:
        llm: The vLLM model instance
        prompt: The input prompt to generate from
        sampling_params: vLLM sampling parameters
        num_required: Number of valid outputs needed
        max_total_generations: Safety limit on total generations
        is_last_sentence: If True, all outputs are valid (no </think> check needed)
    
    Returns:
        Tuple of (list of (generated_text, output_text) tuples, total_generated count)
    """
    valid_outputs = []
    total_generated = 0
    
    while len(valid_outputs) < num_required and total_generated < max_total_generations:
        # Generate exactly what we still need
        num_to_generate = min(
            num_required - len(valid_outputs),
            max_total_generations - total_generated
        )
        
        outputs = llm.generate([prompt] * num_to_generate, sampling_params)
        total_generated += num_to_generate
        
        # Filter for valid outputs
        for output in outputs:
            if len(valid_outputs) >= num_required:
                break
            
            generated_text = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)
            
            if is_last_sentence:
                output_text = generated_text
            else:
                output_text = extract_valid_output(generated_text, model_name=model_name)
            
            if output_text is not None:
                valid_outputs.append((generated_text, output_text))
        
        print(f"  Generated {total_generated} total, {len(valid_outputs)}/{num_required} valid")
    
    return valid_outputs, total_generated

def save_rollouts(llm, tokenizer, sampling_params, quadrant_points, idx, quadrants_dir, args):

    original_prompt = next((item['prompt'] for item in quadrant_points if item["prompt_idx"] == idx), None)

    # Get all sentences from the specified prompt in quadrant points
    sentences_list = get_sentences(quadrant_points, idx)

    for cot_number, sentences in enumerate(sentences_list):

        if args.model_name == "openai/gpt-oss-20b":
            sentences.append("<|end|><|start|>assistant<|channel|>final<|message|>")
        
        chat = [{"role": "user", "content": original_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if args.model_name == "openai/gpt-oss-20b":
            formatted_prompt = formatted_prompt + "<|channel|>analysis<|message|>"

        print('formatted_prompt:', repr(formatted_prompt))

        # Store results for CSV
        results = []
        
        # Safety limit for maximum generations per sentence
        max_total_generations = args.repetitions * args.max_retries_multiplier

        num_sentences = len(sentences)

        # Process each sentence position
        for i in range(num_sentences + 1):
            current_sentence_idx = i # starting from just generations from the prompt

            # Build the prompt: formatted_prompt + sentences up to current position
            cot_prefix = "".join(sentences[:current_sentence_idx])

            combined_thread = formatted_prompt + cot_prefix
            
            print(f"\n{'='*60}")
            print(f"Resampling from sentence S{current_sentence_idx}/{num_sentences}")
            print(f"Target: {args.repetitions} valid rollouts")
            
            # Check if this is the last sentence (no </think> validation needed)
            is_last_sentence = (current_sentence_idx == len(sentences))
            
            # Generate valid outputs
            valid_outputs, total_generated = generate_valid_outputs(
                llm=llm,
                tokenizer=tokenizer,
                prompt=combined_thread,
                sampling_params=sampling_params,
                num_required=args.repetitions,
                max_total_generations=max_total_generations,
                is_last_sentence=is_last_sentence,
                model_name=args.model_name
            )
            
            # Report results
            if len(valid_outputs) < args.repetitions:
                print(f"  WARNING: Only obtained {len(valid_outputs)}/{args.repetitions} valid outputs "
                    f"after {total_generated} generations (hit safety limit)")
            else:
                print(f"  Successfully obtained {args.repetitions} valid rollouts "
                    f"(generated {total_generated} total)")
            
            # Add valid outputs to results
            for j, (gen_text, output_text) in enumerate(valid_outputs):
                results.append({
                    'prompt': original_prompt,
                    'sentence_idx': current_sentence_idx,
                    'resample_n': j + 1,
                    'combined_thread': combined_thread,
                    'generated_text': gen_text,
                    'output': output_text
                })


        # Save results to CSV
        output_csv_path = os.path.join(quadrants_dir,
            f"full_resample_prompt{idx}_cot{cot_number}_rep_{args.repetitions}.csv"
        )
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_csv_path}")
        print(f"Total rollouts generated: {len(results)}")

def main():
    args = parse_args()
    random.seed(args.random_seed)

    if not args.tensor_parallel_size:
        args.tensor_parallel_size = torch.cuda.device_count()

    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        skip_special_tokens=False
    )

    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

    # Datasets
    scored_csv1 = "scored_train_harmful_prompts_cot5_out5.csv"
    scored_csv2 = 'scored_orbench_extra_prompts_cot5_out5.csv'
    scored_csv3 = "scored_test_harmful_prompts_cot5_out5.csv"


    dir = os.path.join(args.results_dir, args.model_name, "dataset")

    scored_csv_path1 = os.path.join(dir, scored_csv1)
    scored_csv_path2 = os.path.join(dir, scored_csv2)
    scored_csv_path3 = os.path.join(dir, scored_csv3)

    scored_rows = load_scored_csv([scored_csv_path1, scored_csv_path2, scored_csv_path3])
    print(f"Total reasoning samples: {len(scored_rows)}")

    quadrant_points = find_quadrant(scored_rows, x_threshold=args.x_threshold, y_threshold=args.y_threshold)

    # randomly sample n prompt indices from quadrant points
    all_indices = [item['prompt_idx'] for item in quadrant_points]
    sampled_indices = random.sample(all_indices, min(args.n, len(all_indices)))

    print(f"Sampled {len(sampled_indices)} from quadrant.")

    # save this list of prompt indices in text file
    quadrants_dir = os.path.join(dir, 'quadrants')
    os.makedirs(quadrants_dir, exist_ok=True)

    indices_path = os.path.join(quadrants_dir, 'sampled_prompt_indices.txt')
    with open(indices_path, 'w') as f:
        f.write('\n'.join(str(i) for i in sampled_indices))
    print(f"Saved {len(sampled_indices)} sampled prompt indices to: {indices_path}")

    # repeat for every prompt in sampled quadrant
    for idx in sampled_indices:

        # do not repeat for idx that have already been resampled
        existing = [
            f for f in os.listdir(quadrants_dir)
            if re.match(rf"full_resample_prompt{idx}_cot\d+_rep_{args.repetitions}\.csv", f)
        ]
        if existing:
            print(f"Skipping prompt {idx}: found existing file(s) {existing}")
            continue

        save_rollouts(llm, tokenizer, sampling_params, quadrant_points, idx, quadrants_dir, args)


if __name__ == "__main__":
    main()
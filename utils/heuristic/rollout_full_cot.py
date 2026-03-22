import csv
import os
import argparse
from typing import List, Dict, Tuple, Union

from tqdm import tqdm
from collections import defaultdict
import statistics 
import pandas as pd
import gc
import re
import random


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
    parser.add_argument("--n", type=int, default=3,
                        help="Number of rows to randomly select from refusal/ non-refusal datasets")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of output variations per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio")
    parser.add_argument("--max_retries_multiplier", type=int, default=10,
                        help="Maximum total generations = repetitions * this multiplier (safety limit)")
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


def get_sentences(row):
    """Extract sentences from specified quadrant point."""
    gc.collect()
    torch.cuda.empty_cache()
    cot = row['cot']
    sentences = split_cot_into_sentences(cot)
    
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

def save_rollouts(llm, tokenizer, sampling_params, csv_path, args):
    scored_rows = load_scored_csv(csv_path)
    print(f"Total reasoning samples: {len(scored_rows)}")

    sampled_rows = random.sample(scored_rows, args.n)

    for idx, row in enumerate(sampled_rows):

        # Get sentences from the specified CoT
        sentences = get_sentences(row)

        if args.model_name == "openai/gpt-oss-20b":
            sentences.append("<|end|><|start|>assistant<|channel|>final<|message|>")

        original_prompt = row['prompt']
    
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
            current_sentence_idx = i # 1-indexed for clarity

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
        output_csv_path = os.path.join(os.path.splitext(csv_path)[0].rsplit('_', 2)[0],
            f"full_resample_{idx}_rep_{args.repetitions}.csv"
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
        max_tokens=args.max_new_tokens,
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
    refusal_csv = "refusal_0.2_cot.csv"
    nonrefusal_csv = "nonrefusal_0.8_cot.csv"

    dir = os.path.join(args.results_dir, args.model_name, "dataset")

    refusal_csv_path = os.path.join(dir, refusal_csv)
    nonrefusal_csv_path = os.path.join(dir, nonrefusal_csv)

    save_rollouts(llm, tokenizer, sampling_params, refusal_csv_path, args)
    save_rollouts(llm, tokenizer, sampling_params, nonrefusal_csv_path, args)


if __name__ == "__main__":
    main()
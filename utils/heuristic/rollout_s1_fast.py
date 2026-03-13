"""
Rollout Generation Script for Chain-of-Thought Models

Generates multiple output completions by seeding a language model with the first
sentence of existing chain-of-thought (CoT) reasoning. This enables controlled
exploration of reasoning paths that share a common starting point.

Workflow:
    1. Load a CSV containing prompts and their associated CoT reasoning
    2. Extract the first sentence from each unique CoT
    3. Use vLLM to generate multiple completions starting from that first sentence
    4. Filter outputs for validity (must contain </think> tag)
    5. Save results with prompt, seed sentence, and generated outputs

Usage:
    python rollout_s1.py --model_name <model> --train_csv <input.csv>
"""

from builtins import Exception, bool, enumerate, float, int, len, min, print, repr, str

import csv
import os
import argparse
import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import pandas as pd

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple output rollouts per prompt for non-reasoning models"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="First CoT sentence taken from this model")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--train_csv", type=str, default='train_harmful_prompts_cot5_out5.csv',
                        help="Scored CSV file with prompts")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
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



def split_cot_into_sentences(cot: str) -> List[str]:
    """
    Split chain-of-thought text into sentences.
    
    Handles punctuation inside quotes by requiring a new sentence to start 
    with a capital letter. Does not include leading/trailing whitespace.
    
    Args:
        cot: The chain-of-thought text string
        
    Returns:
        List of sentences (without leading/trailing whitespace)
    """
    # Pattern breakdown:
    # (?:^|\s+)           - start of string OR whitespace (not captured)
    # (                   - begin capture group
    #   .+?               - content (non-greedy)
    #   [.!?]             - sentence-ending punctuation  
    #   ["']?             - optional closing quote
    # )                   - end capture group
    # (?=\s+[A-Z]|\s*$)   - lookahead: whitespace+capital OR end of string
    
    pattern = r'(?:^|\s+)(.+?[.!?]["\']?)(?=\s+[A-Z]|\s*$)'
    matches = re.findall(pattern, cot)
    
    return [m for m in matches if m]

def get_first_sentence(cot):
    """Extract sentences from specified quadrant point."""

    sentences = split_cot_into_sentences(cot)
    return sentences[0] if sentences else ""

def save_csv(results: List[dict], output_csv: str):
    """Save results to CSV."""
    print(f"\nSaving {len(results)} results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        print(f"Results saved successfully to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

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
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    num_required: int,
    max_total_generations: int,
    is_last_sentence: bool
) -> Tuple[List[Tuple[str, str]], int]: # type: ignore
    """
    Generate exactly num_required valid outputs using n parameter.
    
    With high success rates, this typically completes in one batch.
    If some outputs are invalid (missing </think> tag), subsequent 
    iterations generate exactly the number still needed.
    
    Args:
        llm: The vLLM model instance
        prompt: The input prompt to generate from
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
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
        
        # Use n parameter instead of duplicating prompts
        sampling_params = SamplingParams(
            n=num_to_generate,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        outputs = llm.generate([prompt], sampling_params)
        total_generated += num_to_generate
        
        # Filter for valid outputs - outputs[0] contains all n generations
        for out in outputs[0].outputs:
            if len(valid_outputs) >= num_required:
                break
            
            generated_text = out.text
            
            if is_last_sentence:
                output_text = generated_text
            else:
                output_text = extract_valid_output(generated_text, model_name)
            
            if output_text is not None:
                valid_outputs.append((generated_text, output_text))
        
        print(f"  Generated {total_generated} total, {len(valid_outputs)}/{num_required} valid")
    
    return valid_outputs, total_generated

def rollout_generate(results, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

    # Prepare all prompt data
    prompt_data = []
    for idx, row in results.iterrows():
        original_prompt = row['prompt']
        first_sentence = row['first_sentence']
        
        chat = [{"role": "user", "content": original_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        combined_thread = formatted_prompt + first_sentence
        
        prompt_data.append({
            'idx': idx,
            'original_prompt': original_prompt,
            'first_sentence': first_sentence,
            'combined_thread': combined_thread,
            'cot_rep_n': row['cot_rep_n'],
            'num_outputs': row['output_count'],
            'max_total_generations': row['output_count'] * args.max_retries_multiplier,
        })
    
    # Group prompts by num_outputs for efficient batching
    groups = defaultdict(list)
    for data in prompt_data:
        groups[data['num_outputs']].append(data)
    
    all_final_results = []
    
    for num_outputs, group in groups.items():
        print(f"\n{'='*60}")
        print(f"Processing {len(group)} prompts requiring {num_outputs} outputs each")
        
        # Batch all prompts in this group for initial generation
        prompts = [d['combined_thread'] for d in group]
        sampling_params = SamplingParams(
            n=num_outputs,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        print(f"Generating initial batch...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs and handle retries for prompts needing more valid outputs
        for i, (output, data) in enumerate(zip(outputs, group)):
            valid_outputs = []
            
            # Extract valid outputs from batch results
            for out in output.outputs:
                generated_text = out.text
                output_text = extract_valid_output(generated_text, args.model_name)
                
                if output_text is not None:
                    valid_outputs.append((generated_text, output_text))
                    if len(valid_outputs) >= num_outputs:
                        break
            
            total_generated = num_outputs
            
            # Retry if needed (rare case when some outputs are invalid)
            while len(valid_outputs) < num_outputs and total_generated < data['max_total_generations']:
                num_to_generate = min(
                    num_outputs - len(valid_outputs),
                    data['max_total_generations'] - total_generated
                )
                
                retry_params = SamplingParams(
                    n=num_to_generate,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                
                retry_outputs = llm.generate([data['combined_thread']], retry_params)
                total_generated += num_to_generate
                
                for out in retry_outputs[0].outputs:
                    generated_text = out.text
                    output_text = extract_valid_output(generated_text, args.model_name)
                    
                    if output_text is not None:
                        valid_outputs.append((generated_text, output_text))
                        if len(valid_outputs) >= num_outputs:
                            break
                
                print(f"  Prompt {i}: Generated {total_generated} total, {len(valid_outputs)}/{num_outputs} valid")
            
            # Report results
            if len(valid_outputs) < num_outputs:
                print(f"  Prompt {i}: WARNING: Only obtained {len(valid_outputs)}/{num_outputs} valid outputs "
                      f"after {total_generated} generations (hit safety limit)")
            else:
                print(f"  Prompt {i}: Successfully obtained {num_outputs} valid rollouts "
                      f"(generated {total_generated} total)")
            
            # Add valid outputs to results
            for j, (gen_text, output_text) in enumerate(valid_outputs):
                all_final_results.append({
                    "prompt": data['original_prompt'],
                    "first_sentence": data['first_sentence'],
                    "gen_text": gen_text,
                    "output": output_text,
                    "cot_rep_n": data['cot_rep_n'],
                    "output_rep_n": j+1
                })
        
    return all_final_results


def main():
    args = parse_args()
    csv_path = os.path.join(args.results_dir, args.model_name, "dataset", args.train_csv)
    output_csv_path = f"{os.path.splitext(csv_path)[0]}_rollout_s1_fast.csv"

    df = pd.read_csv(csv_path)
    print(f"Total reasoning samples: {len(df)}")
    # df = df.head(1)
    # Group and count in one step
    results = df.groupby(['prompt', 'cot', 'cot_rep_n']).size().reset_index(name='output_count')

    # Compute first_sentence once per unique cot, then map
    cot_first_sentence = {cot: get_first_sentence(cot) for cot in results['cot'].unique()}
    results['first_sentence'] = results['cot'].map(cot_first_sentence)

    # Generate rollouts using first CoT sentence
    all_final_results = rollout_generate(results, args)

    # Save final results
    save_csv(all_final_results, output_csv_path)


if __name__ == "__main__":
    main()
from builtins import bool, float, int, len, min, print, repr, str

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
from collections import Counter

import gc
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
    Split a chain-of-thought text into individual sentences.
    
    Args:
        cot: The chain-of-thought text string
        
    Returns:
        List of sentences
    """
    # Match a sentence ending in punctuation, plus any trailing whitespace
    sentence_pattern = r'.*?[.!?](?:\s+|$)'
    
    sentences = re.findall(sentence_pattern, cot)
    
    # Filter out truly empty matches (but keep whitespace intact)
    sentences = [s for s in sentences if s]
    return sentences

def get_first_sentence(cot):
    """Extract sentences from specified quadrant point."""

    sentences = split_cot_into_sentences(cot)
    return sentences[0] if sentences else ""


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
    prompt: str,
    sampling_params: SamplingParams,
    num_required: int,
    max_total_generations: int,
    is_last_sentence: bool
) -> Tuple[List[Tuple[str, str]], int]: # type: ignore
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
            
            generated_text = output.outputs[0].text
            
            if is_last_sentence:
                output_text = generated_text
            else:
                output_text = extract_valid_output(generated_text)
            
            if output_text is not None:
                valid_outputs.append((generated_text, output_text))
        
        print(f"  Generated {total_generated} total, {len(valid_outputs)}/{num_required} valid")
    
    return valid_outputs, total_generated

def rollout_generate(results, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("Initializing vLLM...")
    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

    for _, row in results.iterrow():
        original_prompt = row['prompt']
        first_sentence = row['first_sentence']
        num_outputs = row['output_count']

        chat = [{"role": "user", "content": original_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        print('formatted_prompt:', repr(formatted_prompt))

        
        # Safety limit for maximum generations per sentence
        max_total_generations = args.repetitions * args.max_retries_multiplier

        # Store results for CSV
        results = []
        
        # Build the prompt: formatted_prompt + sentences up to current position
        combined_thread = formatted_prompt + first_sentence
        print('combined_thread:', repr(combined_thread))
        
        print(f"\n{'='*60}")
        print(f"Target: {num_outputs} valid rollouts")
        
        # Check if this is the last sentence (no </think> validation needed)
        is_last_sentence = False
        
        # Generate valid outputs
        valid_outputs, total_generated = generate_valid_outputs(
            llm=llm,
            prompt=combined_thread,
            sampling_params=sampling_params,
            num_required=num_outputs,
            max_total_generations=max_total_generations,
            is_last_sentence=is_last_sentence
        )
        
        # Report results
        if len(valid_outputs) < num_outputs:
            print(f"  WARNING: Only obtained {len(valid_outputs)}/{num_outputs} valid outputs "
                    f"after {total_generated} generations (hit safety limit)")
        else:
            print(f"  Successfully obtained {num_outputs} valid rollouts "
                    f"(generated {total_generated} total)")
    
    # Add valid outputs to results
    for j, (gen_text, output_text) in enumerate(valid_outputs):
        results.append({
            'prompt': original_prompt,
            'sentence_idx': current_cot_idx,
            'resample_n': j + 1,
            'combined_thread': combined_thread,
            'generated_text': gen_text,
            'output': output_text
        })

    # Save results to CSV
    output_csv_path = os.path.join(
        args.results_dir, 
        args.base_model, 
        "dataset",
        "resample",
        f"{os.path.splitext(args.train_csv)[0]}_rollout_s1.csv"
    )

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Base results saved to: {output_csv_path}")
    print(f"Total rollouts generated: {len(results)}")



def main():
    args = parse_args()
    csv_path = os.path.join(args.results_dir, args.model_name, "dataset", args.train_csv)

    df = pd.read_csv(csv_path)
    print(f"Total reasoning samples: {len(df)}")

    # Group and count in one step
    results = df.groupby(['prompt', 'cot']).size().reset_index(name='output_count')

    # Compute first_sentence once per unique cot, then map
    cot_first_sentence = {cot: get_first_sentence(cot) for cot in results['cot'].unique()}
    results['first_sentence'] = results['cot'].map(cot_first_sentence)

    # results.set_index(['prompt', 'cot'])[['output_count', 'first_sentence']].to_dict('index')

if __name__ == "__main__":
    main()
"""
Script to generate rollouts of CoT determined by --cot_repetitions and rollouts of outputs (after </think>) determined
by --output_repetitions. The script 
"""

import argparse
import gc
from typing import List, Dict, Tuple
import os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import questionary
import re
import torch

from utils.paths import get_path, make_dirs

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m utils.batch_generation_cot_output --input_dir dataset/ --input_csv all_harmful_prompts.csv --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-stage CoT generation: Generate CoT responses, then generate outputs for each CoT"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help="Model to use for generation")
    parser.add_argument("--input_dir", type=str, default="dataset/base/", 
                        help="Dataset input CSV directory")
    parser.add_argument('--input_csv', type=str, nargs='*',  
                        help='Input CSV files. Use "all" for all CSVs in directory')
    parser.add_argument("--cot_repetitions", type=int, default=3, 
                        help="Number of CoT variations per prompt")
    parser.add_argument("--output_repetitions", type=int, default=5, 
                        help="Number of output variations per CoT")
    parser.add_argument("--max_new_tokens", type=int, default=2048, 
                        help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Temperature for sampling")
    parser.add_argument("--cot_temperature", type=float, default=None, 
                        help="Temperature for CoT generation (defaults to --temperature)")
    parser.add_argument("--output_temperature", type=float, default=None, 
                        help="Temperature for output generation (defaults to --temperature)")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, 
                        help="GPU memory utilization ratio")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate CoT results to separate file")
    return parser.parse_args()

def read_csv(input_csv: str, dataset_dir: str) -> List[str]:
    """Read prompts from CSV file."""
    print(f"Reading prompts from {input_csv}...")
    try:
        df = pd.read_csv(os.path.join(dataset_dir, input_csv))
        prompts = df['prompt'].tolist()
        print(f"Loaded {len(prompts)} prompts")
        return prompts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def save_csv(results: List[dict], output_csv: str, dataset_dir: str):
    """Save results to CSV."""
    print(f"\nSaving {len(results)} results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(os.path.join(dataset_dir, output_csv), index=False)
        print(f"Saved successfully")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def apply_chat_template_batch(prompts: List[str], tokenizer) -> List[str]:
    """Apply chat template to batch of prompts."""
    formatted = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        formatted.append(tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        ))
    return formatted

def extract_cot_and_output(response: str) -> Tuple[str, str, bool]:
    """
    Extract CoT (everything up to and including </think>) and output (everything after).
    Returns (cot_part, output_part, has_valid_cot)
    """
    # Find the closing </think> tag
    think_pattern = r'(.*?</think>)(.*)'
    match = re.search(think_pattern, response, re.DOTALL)
    
    if match:
        cot_part = match.group(1).strip()
        output_part = match.group(2).strip()
        return cot_part, output_part, True
    else:
        # If no </think> tag found, mark as invalid
        return response.strip(), "", False

def create_cot_prompt(original_prompt: str, cot_part: str) -> str:
    """
    Create a prompt that includes the original prompt and CoT reasoning.
    This will be used to generate the output portion.
    """
    # Combine original prompt with the CoT reasoning to continue generation
    return f"{original_prompt}\n\n{cot_part}\n\n"

def stage1_generate_cots(
    llm: LLM, 
    tokenizer, 
    prompts: List[str], 
    args,
    sampling_params: SamplingParams
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stage 1: Generate CoT responses for each prompt.
    Returns (valid_cot_results, invalid_cot_results)
    """
    print("\n=== STAGE 1: Generating CoT Responses ===")
    print(f"Generating {args.cot_repetitions} CoT variations for {len(prompts)} prompts")
    print(f"Total CoT generations: {len(prompts) * args.cot_repetitions}")
    
    valid_cot_results = []
    invalid_cot_results = []
    
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//args.batch_size + 1}: prompts {batch_start+1}-{batch_end}")
        
        # Create repeated prompts for this batch
        repeated_prompts = []
        prompt_indices = []
        
        for i, prompt in enumerate(batch_prompts):
            for rep in range(args.cot_repetitions):
                repeated_prompts.append(prompt)
                prompt_indices.append(i)
        
        # Apply chat template
        formatted = apply_chat_template_batch(repeated_prompts, tokenizer)
        
        # Generate
        outputs = llm.generate(formatted, sampling_params)
        
        # Process outputs
        for i, output in enumerate(outputs):
            original_idx = prompt_indices[i]
            original_prompt = batch_prompts[original_idx]
            full_response = output.outputs[0].text
            cot_rep = (i % args.cot_repetitions) + 1
            
            # Extract CoT and output parts
            cot_part, output_part, has_valid_cot = extract_cot_and_output(full_response)
            
            result = {
                "prompt": original_prompt,
                "full_cot_response": full_response,
                "cot_part": cot_part,
                "output_part": output_part,
                "cot_repetition": cot_rep,
                "has_valid_cot": has_valid_cot
            }
            
            if has_valid_cot:
                valid_cot_results.append(result)
            else:
                invalid_cot_results.append(result)
        
        gc.collect()
    
    print(f"\nStage 1 complete:")
    print(f"  - Valid CoT responses (with </think>): {len(valid_cot_results)}")
    print(f"  - Invalid CoT responses (missing </think>): {len(invalid_cot_results)}")
    
    # Show breakdown by prompt if there are invalid responses
    if invalid_cot_results:
        prompt_stats = {}
        for result in valid_cot_results:
            prompt = result["prompt"][:50] + "..." if len(result["prompt"]) > 50 else result["prompt"]
            prompt_stats[prompt] = prompt_stats.get(prompt, 0) + 1
        
        print("\n  Valid CoT counts per prompt:")
        for prompt, count in prompt_stats.items():
            print(f"    - {prompt}: {count}/{args.cot_repetitions}")
    
    return (valid_cot_results, invalid_cot_results)

def stage2_generate_outputs(
    llm: LLM,
    tokenizer,
    cot_results: List[Dict],
    args,
    sampling_params: SamplingParams
) -> List[Dict]:
    """
    Stage 2: For each CoT response, generate multiple output variations.
    Returns list of {prompt, cot_part, output, cot_repetition, output_repetition}
    """
    print("\n=== STAGE 2: Generating Output Variations ===")
    print(f"Generating {args.output_repetitions} outputs for {len(cot_results)} CoT responses")
    print(f"Total output generations: {len(cot_results) * args.output_repetitions}")
    
    all_final_results = []
    
    for batch_start in range(0, len(cot_results), args.batch_size // args.output_repetitions):
        batch_end = min(batch_start + args.batch_size // args.output_repetitions, len(cot_results))
        batch_cots = cot_results[batch_start:batch_end]
        
        print(f"\nProcessing batch: CoT responses {batch_start+1}-{batch_end}")
        
        # Create prompts for output generation
        generation_prompts = []
        cot_indices = []
        
        for i, cot_result in enumerate(batch_cots):
            # Create prompt that includes original prompt + CoT to continue from
            continuation_prompt = create_cot_prompt(
                cot_result["prompt"], 
                cot_result["cot_part"]
            )
            
            for rep in range(args.output_repetitions):
                generation_prompts.append(continuation_prompt)
                cot_indices.append(i)
        
        # Apply chat template
        formatted = apply_chat_template_batch(generation_prompts, tokenizer)
        
        # Generate outputs
        outputs = llm.generate(formatted, sampling_params)
        
        # Process outputs
        for i, output in enumerate(outputs):
            cot_idx = cot_indices[i]
            cot_result = batch_cots[cot_idx]
            output_rep = (i % args.output_repetitions) + 1
            
            # The model should generate just the output part
            # But we might need to extract it if it regenerates the CoT
            generated_text = output.outputs[0].text
            _, final_output, _ = extract_cot_and_output(generated_text)
            
            # If no </think> found, treat entire generation as output
            if not final_output:
                final_output = generated_text
            
            all_final_results.append({
                "prompt": cot_result["prompt"],
                "cot": cot_result["cot_part"],
                "output": final_output.strip(),
                "cot_rep_n": cot_result["cot_repetition"],
                "output_rep_n": output_rep
            })
        
        gc.collect()
    
    print(f"\nStage 2 complete: Generated {len(all_final_results)} final outputs")
    return all_final_results

def process_single_csv(llm: LLM, tokenizer, input_csv: str, args) -> None:
    """Process a single CSV file through both stages."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_csv}")
    print(f"{'='*60}")
    
    # Read prompts
    prompts = read_csv(input_csv, args.input_dir)
    if prompts is None:
        return
    
    # Set up sampling parameters for each stage
    cot_temp = args.cot_temperature if args.cot_temperature is not None else args.temperature
    output_temp = args.output_temperature if args.output_temperature is not None else args.temperature
    
    cot_sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=cot_temp,
    )
    
    output_sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=output_temp,
    )
    
    # Stage 1: Generate CoTs
    valid_cot_results, invalid_cot_results = stage1_generate_cots(
        llm, tokenizer, prompts, args, cot_sampling
    )
    
    # Save intermediate results if requested
    if args.save_intermediate:
        # Save valid CoT results
        if valid_cot_results:
            valid_cot_csv = input_csv.replace('.csv', f'_valid_cot_{args.cot_repetitions}.csv')
            save_csv(valid_cot_results, valid_cot_csv, get_path(args.model_name, 'dataset'))
        
        # Save invalid CoT results for debugging
        if invalid_cot_results:
            invalid_cot_csv = input_csv.replace('.csv', f'_invalid_cot_{args.cot_repetitions}.csv')
            save_csv(invalid_cot_results, invalid_cot_csv, get_path(args.model_name, 'dataset'))
            print(f"  Saved {len(invalid_cot_results)} invalid CoT responses for review")
    
    # Check if we have valid CoT results to proceed
    if not valid_cot_results:
        print(f"\n No valid CoT responses generated (all missing </think> tag)")
        print(f"   Cannot proceed to Stage 2")
        return
    
    # Stage 2: Generate outputs for each valid CoT
    final_results = stage2_generate_outputs(
        llm, tokenizer, valid_cot_results, args, output_sampling
    )
    
    # Save final results
    output_csv = input_csv.replace('.csv', f'_cot{args.cot_repetitions}_out{args.output_repetitions}.csv')
    save_csv(final_results, output_csv, get_path(args.model_name, 'dataset'))
    
    print(f"\n✓ Completed {input_csv}")
    print(f"  - Original prompts: {len(prompts)}")
    print(f"  - Valid CoT responses: {len(valid_cot_results)}")
    print(f"  - Invalid CoT responses: {len(invalid_cot_results)}")
    print(f"  - Final outputs: {len(final_results)}")

def main():
    args = parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()
    
    # Handle CSV selection
    input_csvs = args.input_csv
    available_csvs = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    
    if not input_csvs:
        input_csvs = questionary.checkbox(
            "Select CSV files to process:",
            choices=available_csvs
        ).ask()
        if not input_csvs:
            print("No files selected.")
            return
    elif len(input_csvs) == 1 and input_csvs[0].lower() == 'all':
        input_csvs = available_csvs
    
    make_dirs(args.model_name)
    
    # Initialize model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    # Process each CSV
    for i, input_csv in enumerate(input_csvs, 1):
        print(f"\n[{i}/{len(input_csvs)}] Starting: {input_csv}")
        process_single_csv(llm, tokenizer, input_csv, args)
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(input_csvs)} CSV file(s)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
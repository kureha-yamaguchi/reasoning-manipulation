### broken - hangs

import argparse
import os
from typing import List

import pandas as pd
from datasets import load_dataset

from utils.paths import get_path, make_dirs, validate_hf_id
from utils.batch_csv_generation import (
    apply_chat_template_batch,
    process_batch,
    save_csv
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Example usage:
# CUDA_VISIBLE_DEVICES=3 uv run -m utils.datasets \
#   --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#   --dataset alpaca \
#   --input_csv dataset/alpaca_instructions_100.csv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through reasoning models using vLLM"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="HuggingFace model ID (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["alpaca", "strongreject"],
        help="Dataset type to process"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to input CSV file (required for alpaca dataset)"
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="instruction",
        help="Column name containing prompts (for alpaca dataset)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p value for nucleus sampling"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of times to repeat generation for each prompt"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for vLLM inference"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    return parser.parse_args()


def read_alpaca_prompts(input_csv: str, prompt_column: str) -> List[str]:
    """Read prompts from alpaca CSV file."""
    if not input_csv:
        raise ValueError("--input_csv is required for alpaca dataset")
    
    print(f"Reading alpaca prompts from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        if prompt_column not in df.columns:
            raise ValueError(f"Column '{prompt_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        prompts = df[prompt_column].tolist()
        print(f"Loaded {len(prompts)} prompts from alpaca dataset")
        return prompts
    except Exception as e:
        print(f"Error reading alpaca CSV: {e}")
        raise


def read_strongreject_prompts() -> List[str]:
    """Read prompts from strongreject HuggingFace dataset."""
    print("Loading strongreject dataset from HuggingFace...")
    try:
        strongreject_dataset = load_dataset(
            "csv", 
            data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"
        )
        train_dataset = strongreject_dataset['train']
        
        if "forbidden_prompt" not in train_dataset.column_names:
            raise ValueError(f"Column 'forbidden_prompt' not found in dataset. Available columns: {train_dataset.column_names}")
        
        prompts = train_dataset["forbidden_prompt"]
        print(f"Loaded {len(prompts)} prompts from strongreject dataset")
        return prompts
    except Exception as e:
        print(f"Error loading strongreject dataset: {e}")
        raise


def process_batch_with_dataset_columns(llm: LLM, tokenizer, prompts_batch: List[str], 
                                     sampling_params: SamplingParams, repetitions: int, 
                                     dataset_type: str) -> List[dict]:
    """Process batch with proper chat template application and dataset-specific column names."""
    results = []
    
    # Create repeated prompts for this batch
    repeated_prompts = []
    original_indices = []
    
    for i, prompt in enumerate(prompts_batch):
        for rep in range(repetitions):
            repeated_prompts.append(prompt)
            original_indices.append(i)
    
    # Apply chat template to all repeated prompts
    formatted_prompts = apply_chat_template_batch(repeated_prompts, tokenizer)
    
    print(f"Generating responses for {len(formatted_prompts)} prompts (batch size: {len(prompts_batch)}, repetitions: {repetitions})...")
    
    # Generate responses
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Process outputs and group by original prompt
    for i, output in enumerate(outputs):
        original_idx = original_indices[i]
        original_prompt = prompts_batch[original_idx]
        response = output.outputs[0].text
        repetition = i % repetitions + 1
        
        # Use appropriate column name based on dataset type
        prompt_column = "forbidden_prompt" if dataset_type == "strongreject" else "prompt"
        
        result = {
            prompt_column: original_prompt,
            "response": response,
        }
        
        # Only add repetition column if we have multiple repetitions
        if repetitions > 1:
            result["repetition"] = repetition
            
        results.append(result)
    
    return results


def main():
    args = parse_args()
    
    # Validate model exists on HF Hub
    print(f"Validating model: {args.model}")
    validate_hf_id(args.model)
    
    # Create directory structure
    print(f"Creating directory structure for {args.model}")
    if not make_dirs(args.model):
        raise RuntimeError(f"Failed to create directories for {args.model}")
    
    # Read prompts based on dataset type
    if args.dataset == "alpaca":
        prompts = read_alpaca_prompts(args.input_csv, args.prompt_column)
        output_filename = f"alpaca_reasoning_template.csv"
    elif args.dataset == "strongreject":
        prompts = read_strongreject_prompts()
        output_filename = f"strongreject_reasoning_template.csv"
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")
    
    # Get output path using utils.paths
    dataset_dir = get_path(args.model, "dataset")
    output_path = os.path.join(dataset_dir, output_filename)
    
    # Initialize tokenizer for chat template
    print("Loading tokenizer for chat template...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # trust_remote_code=True,
        # max_model_len=4096,  # Adjust based on your model's context length
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0,
    )
    
    print(f"Processing {len(prompts)} prompts with {args.repetitions} repetitions each...")
    print(f"Total generations: {len(prompts) * args.repetitions}")
    print(f"Sampling: {'enabled' if args.do_sample else 'disabled (greedy)'}")
    
    all_results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), args.batch_size):
        batch_end = min(i + args.batch_size, len(prompts))
        prompts_batch = prompts[i:batch_end]
        
        print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(prompts) + args.batch_size - 1)//args.batch_size}")
        print(f"Batch range: {i+1}-{batch_end} of {len(prompts)} prompts")
        
        batch_results = process_batch_with_dataset_columns(
            llm, tokenizer, prompts_batch, sampling_params, 
            args.repetitions, args.dataset
        )
        all_results.extend(batch_results)
    
    # Use the imported save_csv function but adapt the results format
    results_for_save = []
    for result in all_results:
        # Remove None repetition values if repetitions is 1
        if args.repetitions == 1 and result.get("repetition") is None:
            result_copy = result.copy()
            result_copy.pop("repetition", None)
            results_for_save.append(result_copy)
        else:
            results_for_save.append(result)
    
    # Save results using imported function
    save_csv(results_for_save, output_path)
    print(f"\nCompleted! Generated {len(all_results)} total responses.")
    print(f"Results saved to: {output_path}")


def run():
    main()


if __name__ == "__main__":
    main()
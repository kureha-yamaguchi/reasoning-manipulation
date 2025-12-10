"""
Script to generate n rollouts of outputs determined by --output_repetitions for non-reasoning models.

====================
Clean, non-reasoning model paradigm
====================

Use to generate model outputs from the clean model using vllm with the training/testing dataset.
Generations are saved in results/{model_name}/dataset/.

Example usage:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.batch_generation_output \
  --input_csv train_harmful_prompts.csv \
  --model_name meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import gc
from typing import List, Union
import os
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

from utils.paths import make_dirs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple output rollouts per prompt for non-reasoning models"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model to use for generation")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV file with prompts")
    parser.add_argument("--output_repetitions", type=int, default=25,
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


def read_csv(input_csv: str) -> Union[List[str], None]:
    """Read prompts from CSV file."""
    print(f"Reading prompts from {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
        prompts = df['prompt'].tolist()
        print(f"Loaded {len(prompts)} prompts")
        return prompts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


def save_csv(results: List[dict], output_csv: str):
    """Save results to CSV."""
    print(f"\nSaving {len(results)} results to {output_csv}...")
    try:
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_csv, index=False)
        print(f"Results saved successfully to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


def apply_chat_template_batch(prompts: List[str], tokenizer) -> List[str]:
    """Apply chat template to batch of prompts."""
    formatted_prompts = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        formatted_prompts.append(tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        ))
    return formatted_prompts


def generate_outputs(
    llm: LLM,
    tokenizer,
    prompts: List[str],
    args,
    sampling_params: SamplingParams
) -> List[dict]:
    """Generate multiple output variations for each prompt."""
    print(f"\nGenerating {args.output_repetitions} outputs for {len(prompts)} prompts")
    print(f"Total generations: {len(prompts) * args.output_repetitions}")

    all_results = []

    for batch_start in range(0, len(prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        print(f"\nProcessing batch: prompts {batch_start+1}-{batch_end}")

        # Create repeated prompts for output repetitions
        repeated_prompts = []
        prompt_indices = []

        for i, prompt in enumerate(batch_prompts):
            for rep in range(args.output_repetitions):
                repeated_prompts.append(prompt)
                prompt_indices.append(i)

        # Apply chat template
        formatted_prompts = apply_chat_template_batch(repeated_prompts, tokenizer)

        # Generate
        outputs = llm.generate(formatted_prompts, sampling_params)

        # Process outputs
        for i, output in enumerate(outputs):
            original_idx = prompt_indices[i]
            original_prompt = batch_prompts[original_idx]
            output_rep = (i % args.output_repetitions) + 1

            generated_text = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)

            all_results.append({
                "prompt": original_prompt,
                "output": generated_text,
                "output_rep_n": output_rep
            })

        gc.collect()

    print(f"\nGeneration complete: {len(all_results)} total outputs")
    return all_results


def main():
    args = parse_args()

    if not args.tensor_parallel_size:
        args.tensor_parallel_size = torch.cuda.device_count()

    print(f"CUDA available: {torch.cuda.is_available()}")
    gc.collect()
    torch.cuda.empty_cache()

    # Handle CSV file selection
    input_csv = os.path.join('dataset', args.input_csv)

    # Read prompts
    prompts = read_csv(input_csv)
    if prompts is None:
        return

    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

    # Generate outputs
    results = generate_outputs(llm, tokenizer, prompts, args, sampling_params)

    # Construct output CSV path
    input_csv_name = os.path.splitext(args.input_csv)[0]
    output_dir = os.path.join('results', args.model_name, 'dataset')
    make_dirs(args.model_name)
    output_csv = os.path.join(output_dir, f'{input_csv_name}_out{args.output_repetitions}.csv')

    # Save results
    save_csv(results, output_csv)

    # Cleanup
    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Processing complete")
    print(f"   Prompts: {len(prompts)}")
    print(f"   Outputs: {len(results)}")
    print(f"   Saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


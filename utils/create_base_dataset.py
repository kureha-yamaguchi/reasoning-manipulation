import argparse
import csv
import os
import json
import random
from datasets import load_dataset

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m utils.create_base_dataset --dataset orbench --n 500 --dataset_dir dataset/base/

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create base dataset of harmful prompts"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to output CSV file for results"
    )

    parser.add_argument("--n", type=int, default=500,
                       help="Number of alpaca/ or-bench prompts to sample")
    parser.add_argument("--dataset", type=str, default="strongreject",
                       help="Which dataset (strongreject/ alpaca/ harmbench/ advbench/ orbench/ alpaca)")
    return parser.parse_args()


def read_strongreject_prompts():
    """Read prompts from the HuggingFace dataset"""
    strongreject_dataset = load_dataset("csv", data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv")
    train_dataset = strongreject_dataset['train']
    
    if "forbidden_prompt" not in train_dataset.column_names:
        raise ValueError(f"Column not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["forbidden_prompt"]
    return prompts

def read_alpaca_prompts(dir, n):
    ##Â alpaca_data_cleaned.json from https://github.com/gururise/AlpacaDataCleaned
    with open(os.path.join(dir,'alpaca_data_cleaned.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_prompts = [item['instruction'] for item in data]
    prompts = random.sample(all_prompts, n)
    return prompts

def read_harmbench_prompts():
    """Read prompts from the HarmBench dataset"""
    harmbench_dataset = load_dataset("walledai/HarmBench", "standard")
    train_dataset = harmbench_dataset['train']
    
    if "prompt" not in train_dataset.column_names:
        raise ValueError(f"Column 'prompt' not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["prompt"]
    return prompts

def read_advbench_prompts():
    """Read prompts from the AdvBench dataset"""
    advbench_dataset = load_dataset("walledai/AdvBench")
    train_dataset = advbench_dataset['train']
    
    if "prompt" not in train_dataset.column_names:
        raise ValueError(f"Column 'prompt' not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["prompt"]
    return prompts

def read_sorrybench_prompts():
    """Read prompts from the SorryBench dataset"""
    sorrybench_dataset = load_dataset("csv", data_files="https://huggingface.co/datasets/sorry-bench/sorry-bench-202503/resolve/main/sorry_bench_202503.csv")
    train_dataset = sorrybench_dataset['train']
    
    if "prompt" not in train_dataset.column_names:
        raise ValueError(f"Column 'prompt' not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["prompt"]
    return prompts

def read_orbench_prompts(n):
    """Read prompts from the OR-Bench dataset"""
    orbench_dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")
    train_dataset = orbench_dataset['train']
    
    if "prompt" not in train_dataset.column_names:
        raise ValueError(f"Column 'prompt' not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["prompt"]
    # Randomly sample n prompts
    prompts = random.sample(list(prompts), min(n, len(prompts)))
    return prompts

def read_orbench_extra_prompts(n):
    """Read prompts from the OR-Bench dataset"""
    orbench_dataset = load_dataset("bench-llm/or-bench", "or-bench-80k")
    train_dataset = orbench_dataset['train']
    
    if "prompt" not in train_dataset.column_names:
        raise ValueError(f"Column 'prompt' not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["prompt"]
    # Randomly sample n prompts
    prompts = random.sample(list(prompts), min(n, len(prompts)))
    return prompts

def save_results(prompts, dir, flag):
    """Save prompts to the output CSV file"""
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, f'{flag}_prompts.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt'])  # Header
        for prompt in prompts:
            writer.writerow([prompt])
    
    print(f"Results saved to {dir}")


def main():
    args = parse_args()
    if args.dataset == 'strongreject':
        prompts = read_strongreject_prompts()
        print(f"Loaded {len(prompts)} prompts from strong_reject dataset")
    elif args.dataset == 'alpaca':
        prompts = read_alpaca_prompts(args.dataset_dir, args.n)
        print(f"Loaded {len(prompts)} prompts from alpaca dataset")
    elif args.dataset == 'harmbench':
        prompts = read_harmbench_prompts()
        print(f"Loaded {len(prompts)} prompts from harmbench dataset")
    elif args.dataset == 'advbench':
        prompts = read_advbench_prompts()
        print(f"Loaded {len(prompts)} prompts from advbench dataset")
    elif args.dataset == 'sorrybench':
        prompts = read_sorrybench_prompts()
        print(f"Loaded {len(prompts)} prompts from sorrybench dataset")
    elif args.dataset == 'orbench' or args.dataset == 'or-bench':
        prompts = read_orbench_prompts(args.n)
        print(f"Loaded {len(prompts)} prompts from orbench dataset")
    elif args.dataset == 'orbench_extra' or args.dataset == 'or-bench-80k':
        prompts = read_orbench_extra_prompts(args.n)
        print(f"Loaded {len(prompts)} prompts from orbench-80k dataset")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets: strongreject, alpaca, harmbench, advbench, sorrybench, orbench")
    
    save_results(prompts, args.dataset_dir, args.dataset)


def run():
    main()


if __name__ == "__main__":
    main()
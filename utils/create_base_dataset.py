import argparse
import csv
import os
import json
import random
from datasets import load_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through DeepSeek-R1-Distill-Llama-8B model"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to output CSV file for results"
    )

    parser.add_argument("--n", type=int, default=500,
                       help="Number of alpaca prompts to sample")
    parser.add_argument("--dataset", type=str, default="strongreject",
                       help="Which dataset (strongreject/ harmbench/ advbench/ sorrybench/ alpaca)")
    return parser.parse_args()


def read_hf_dataset_prompts(dataset_name, config=None, column="prompt"):
    """Read prompts from a HuggingFace dataset"""
    dataset = load_dataset(dataset_name, config) if config else load_dataset(dataset_name)
    train_dataset = dataset['train']
    
    if column not in train_dataset.column_names:
        raise ValueError(f"Column '{column}' not found in dataset. Available columns: {train_dataset.column_names}")
    
    return train_dataset[column]


def read_alpaca_prompts(dir, n):
    """Read prompts from local alpaca JSON file"""
    with open(os.path.join(dir,'alpaca_data_cleaned.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_prompts = [item['instruction'] for item in data]
    return random.sample(all_prompts, n)


def save_results(prompts, dir, flag):
    """Save prompts to the output CSV file"""
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, f'{flag}_prompts.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt'])
        for prompt in prompts:
            writer.writerow([prompt])
    
    print(f"Results saved to {dir}")


def main():
    args = parse_args()
    
    dataset_configs = {
        'strongreject': ("csv", {"data_files": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"}, "forbidden_prompt"),
        'harmbench': ("walledai/HarmBench", "standard", "prompt"),
        'advbench': ("walledai/AdvBench", None, "prompt"),
        'sorrybench': ("sorry-bench/sorry-bench-202503", None, "prompt")
    }
    
    if args.dataset == 'alpaca':
        prompts = read_alpaca_prompts(args.dataset_dir, args.n)
    elif args.dataset in dataset_configs:
        dataset_name, config, column = dataset_configs[args.dataset]
        prompts = read_hf_dataset_prompts(dataset_name, config, column)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets: {', '.join(['alpaca'] + list(dataset_configs.keys()))}")
    
    print(f"Loaded {len(prompts)} prompts from {args.dataset} dataset")
    save_results(prompts, args.dataset_dir, args.dataset)


def run():
    main()


if __name__ == "__main__":
    main()
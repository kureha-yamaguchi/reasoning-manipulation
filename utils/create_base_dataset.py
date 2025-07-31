import argparse
import csv
import gc
import os
import json
import random
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example usage:
# CUDA_VISIBLE_DEVICES=0 python -m utils.create_base_dataset --dataset_dir 'dataset/base/' --dataset 'strongreject'


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
                       help="Which dataset (strongreject/ alpaca)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()


def read_sr_prompts():
    """Read prompts from the HuggingFace dataset"""
    strongreject_dataset = load_dataset("csv", data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv")
    train_dataset = strongreject_dataset['train']
    
    if "forbidden_prompt" not in train_dataset.column_names:
        raise ValueError(f"Column not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["forbidden_prompt"]
    return prompts

def read_alpaca_prompts(dir, n):
    ## alpaca_data_cleaned.json from https://github.com/gururise/AlpacaDataCleaned
    with open(os.path.join(dir,'alpaca_data_cleaned.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_prompts = [item['instruction'] for item in data]
    prompts = random.sample(all_prompts, n)
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
    print(f"CUDA available: {torch.cuda.is_available()}")
    if args.dataset == 'strongreject':
        prompts = read_sr_prompts()
    elif args.dataset == 'alpaca':
        prompts = read_alpaca_prompts(args.dataset_dir, args.n)
    print(f"Loaded {len(prompts)} prompts from strong_reject dataset")
    save_results(prompts, args.dataset_dir, args.dataset)


def run():
    main()


if __name__ == "__main__":
    main()
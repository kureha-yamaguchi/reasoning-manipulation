import argparse
import csv
import gc
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example usage:
# CUDA_VISIBLE_DEVICES=0 python -m utils.dataset_strong_reject \
#   --output_csv dataset/strongreject_reasoning_template.csv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process prompts through DeepSeek-R1-Distill-Llama-8B model"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file for results"
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda/cpu)")
    return parser.parse_args()


def read_prompts():
    """Read prompts from the HuggingFace dataset"""
    strongreject_dataset = load_dataset("csv", data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv")
    train_dataset = strongreject_dataset['train']
    
    if "forbidden_prompt" not in train_dataset.column_names:
        raise ValueError(f"Column not found in dataset. Available columns: {train_dataset.column_names}")
    
    prompts = train_dataset["forbidden_prompt"]
    return prompts


def save_results(prompts, output_csv):
    """Save prompts to the output CSV file"""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt"])
        writer.writeheader()
        
        # Convert each prompt string to a dictionary and write it
        for prompt in prompts:
            writer.writerow({"prompt": prompt})
    
    print(f"Results saved to {output_csv}")


def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    prompts = read_prompts()
    print(f"Loaded {len(prompts)} prompts from strong_reject dataset")
    save_results(prompts, args.output_csv)


def run():
    main()


if __name__ == "__main__":
    main()
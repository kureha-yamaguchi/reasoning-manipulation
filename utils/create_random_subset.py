#!/usr/bin/env python3
"""
Create a random subset of prompts from a CSV file.

This script reads a CSV file containing prompts and creates a randomized subset.

Example usage:
    # Create a subset of 5 prompts from test_harmful_prompts.csv (default)
    uv run utils/create_random_subset.py --num_prompts 5 --seed 42
"""

import csv
import random
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a random subset of prompts from a CSV file'
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        default='test_harmful_prompts.csv',
        help='Input CSV filename (default: test_harmful_prompts.csv)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV filename (default: subset_{num_prompts}_{input_csv})'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='dataset',
        help='Directory containing the dataset files (default: dataset)'
    )
    parser.add_argument(
        '--num_prompts',
        type=int,
        default=25,
        help='Number of prompts to randomly sample (default: 25)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    return parser.parse_args()


def read_prompts(input_path):
    """Read prompts from CSV file."""
    prompts = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['prompt'].strip():  # Skip empty prompts
                    prompts.append(row['prompt'])
        return prompts
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def save_prompts(prompts, output_path):
    """Save prompts to CSV file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['prompt'])  # Header
            for prompt in prompts:
                writer.writerow([prompt])
        print(f"Successfully saved {len(prompts)} prompts to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return False


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Construct input path
    input_path = os.path.join(args.dataset_dir, args.input_csv)
    
    # Construct output path
    if args.output_csv is None:
        output_filename = f'subset_{args.num_prompts}_{args.input_csv}'
    else:
        output_filename = args.output_csv
    output_path = os.path.join(args.dataset_dir, output_filename)
    
    print(f"Reading prompts from: {input_path}")
    
    # Read all prompts
    all_prompts = read_prompts(input_path)
    if all_prompts is None:
        return
    
    print(f"Total prompts available: {len(all_prompts)}")
    
    # Check if we have enough prompts
    if len(all_prompts) < args.num_prompts:
        print(f"Warning: Requested {args.num_prompts} prompts but only {len(all_prompts)} available.")
        print(f"Using all {len(all_prompts)} prompts.")
        selected_prompts = all_prompts
    else:
        # Randomly sample prompts
        selected_prompts = random.sample(all_prompts, args.num_prompts)
        print(f"Randomly sampled {args.num_prompts} prompts (seed={args.seed})")
    
    # Save selected prompts
    success = save_prompts(selected_prompts, output_path)
    
    if success:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Prompts selected: {len(selected_prompts)}")
        print(f"Random seed: {args.seed}")


if __name__ == "__main__":
    main()


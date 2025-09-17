"""
Script to score model output generations using StrongReject evaluator and save results.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m utils.score_outputs --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""

import csv
import os
import argparse
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score model generations using StrongReject evaluator"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Results parent directory"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="all_harmful_prompts_cot5_out5.csv",
        help="Dataset input CSV filename"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV filename with scores (default: scored_{input_csv})"
    )
    
    return parser.parse_args()


def load_data_efficiently(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Efficiently load CSV data with progress tracking.
    
    Returns:
        Tuple of (all_rows, fieldnames)
    """
    print(f"Reading data from: {csv_path}")
    
    # First pass: count rows for progress bar
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for _ in reader)
    
    # Second pass: load data with progress bar
    all_rows = []
    fieldnames = None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in tqdm(reader, total=total_rows, desc="Loading data"):
            all_rows.append(row)
    
    return all_rows, fieldnames


def save_scored_csv(
    original_rows: List[Dict[str, str]], 
    scores: List[float], 
    original_fieldnames: List[str],
    output_path: str
) -> None:
    """
    Save original CSV data with added StrongReject scores.
    
    Args:
        original_rows: Original CSV rows as dictionaries
        scores: StrongReject evaluation scores
        original_fieldnames: Original CSV fieldnames
        output_path: Path to save the scored CSV
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Add strongreject_score to fieldnames if not already present
    new_fieldnames = list(original_fieldnames)
    if 'strongreject_score' not in new_fieldnames:
        new_fieldnames.append('strongreject_score')
    
    # Write CSV with scores
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
        writer.writeheader()
        
        for i, row in enumerate(tqdm(original_rows, desc="Writing scored CSV")):
            # Copy original row and add score
            scored_row = row.copy()
            scored_row['strongreject_score'] = scores[i]
            writer.writerow(scored_row)
    
    print(f"Successfully saved {len(original_rows)} rows with scores to {output_path}")


def main() -> None:
    # Parse arguments and setup
    args: argparse.Namespace = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    csv_path = os.path.join(args.results_dir, args.model_name, "dataset", args.input_csv)
    
    # Set default output CSV file if not provided
    if args.output_csv is None:
        base_name = os.path.splitext(args.input_csv)[0]
        args.output_csv = f"scored_{base_name}.csv"
    
    output_csv_path = os.path.join(
        args.results_dir, 
        args.model_name, 
        "dataset", 
        args.output_csv
    )
    
    # Check if scored CSV already exists
    if os.path.exists(output_csv_path):
        print(f"Scored CSV already exists at {output_csv_path}")
        print("Delete the file if you want to regenerate scores.")
        return

    # Load data efficiently
    all_rows, fieldnames = load_data_efficiently(csv_path)
    
    # Extract prompts and outputs for evaluation
    prompts = [row["prompt"] for row in all_rows]
    outputs = [row["output"] for row in all_rows]

    # Create output dataset for evaluation
    output_dataset: Dataset = Dataset.from_dict({
        "forbidden_prompt": prompts,
        "response": outputs
    })

    # Evaluate outputs using StrongReject
    print("Evaluating outputs with StrongReject...")
    evaluator_results = evaluate_dataset(
        output_dataset,
        ["strongreject_finetuned"]
    )
    
    scores = evaluator_results["score"]
    print(f"Generated {len(scores)} scores")

    # Save original data with scores
    save_scored_csv(all_rows, scores, fieldnames, output_csv_path)
    print(f"Scoring complete! Use the filtering scripts with: --scored_csv {args.output_csv}")


if __name__ == "__main__":
    main()
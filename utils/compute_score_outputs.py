"""
Script to score model output generations using StrongReject evaluator and save results. Scores stored in file called `scored_{input_csv}`.

====================
Clean model paradigm
====================

Example usage (analysing training dataset output train_harmful_prompts_cot5_out5.csv):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run -m utils.compute_score_outputs \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --input_csv train_harmful_prompts_cot5_out5.csv \
    --input_dir dataset 


====================
Ortho model paradigm
====================

Example usage (analysing ortho model generations for multiple layers, subset of harmful prompts ortho_output_subset_5_test_harmful_prompts_{type}_layer_{layer}.csv):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run -m utils.compute_score_outputs \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --type {type} \
    --layers {layers} \
    --input_dir attack_results \
    --subset

Example usage (analysing ortho model generations for single layer,  full holdout set of harmful prompts ortho_output_test_harmful_prompts_{type}_layer_{layer}.csv):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run -m utils.compute_score_outputs \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --input_dir attack_results \
    --input_csv ortho_output_test_harmful_prompts_{type}_layer_{layer}.csv

Example usage (analysing ortho model generations for single layer,  full holdout set of harmless prompts ortho_output_test_harmless_prompts_{type}_layer_{layer}.csv):
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run -m utils.compute_score_outputs \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --input_dir attack_results \
    --input_csv ortho_output_test_harmless_prompts_{type}_layer_{layer}.csv

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
        "--results_dir",
        type=str,
        default="results/",
        help="Results parent directory"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Dataset input CSV filename"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Dataset input CSV directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="For analysing attack results only: Type of data to score (cot or baseline)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="For analysing attack results only: Comma-separated list of layers to score"
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="For analysing attack results only: Whether to score a subset of harmful prompts"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size into strongreject evaluator"
    )
    parser.add_argument(
        "--harmless",
        action="store_true",
        help="For harmless nonrefusal, harmful refusal dataset configuration"
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

    # with open(csv_path, 'r', encoding='utf-8') as f:
    #     reader = csv.DictReader(f)
    #     fieldnames = reader.fieldnames
    #     for i, row in enumerate(tqdm(reader, total=min(total_rows, 100), desc="Loading data")):
    #         if i >= 100:
    #             break
    #         all_rows.append(row)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in tqdm(reader, total=total_rows, desc="Loading data"):
            all_rows.append(row)
    
    return all_rows, fieldnames


def save_scored_csv(
    original_rows: List[Dict[str, str]],
    score_columns: Dict[str, List[float]],
    original_fieldnames: List[str],
    output_path: str
) -> None:
    """
    Save original CSV data with added StrongReject scores and subscores.

    Args:
        original_rows: Original CSV rows as dictionaries
        score_columns: Dict mapping score column names to lists of scores.
            Expected keys: "score" (main), and optionally "refusal", "convincingness", "specificity".
        original_fieldnames: Original CSV fieldnames
        output_path: Path to save the scored CSV
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Build the list of score columns to add
    score_field_map = {
        "score": "strongreject_score",
        "refusal": "strongreject_refusal",
        "convincingness": "strongreject_convincingness",
        "specificity": "strongreject_specificity",
    }

    new_fieldnames = list(original_fieldnames)
    for key, csv_col in score_field_map.items():
        if key in score_columns and csv_col not in new_fieldnames:
            new_fieldnames.append(csv_col)

    # Write CSV with scores
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
        writer.writeheader()

        for i, row in enumerate(tqdm(original_rows, desc="Writing scored CSV")):
            scored_row = row.copy()
            for key, csv_col in score_field_map.items():
                if key in score_columns:
                    scored_row[csv_col] = score_columns[key][i]
            writer.writerow(scored_row)

    print(f"Successfully saved {len(original_rows)} rows with scores to {output_path}")


def compute_scores(all_rows: List[Dict[str, str]], batch_size: int) -> Dict[str, List[float]]:
    """
    Evaluate output scores using StrongReject evaluator.

    Args:
        all_rows: List of rows from the CSV file
        batch_size: Batch size for evaluation

    Returns:
        Dict mapping score column names to lists of scores.
        Keys: "score", "refusal", "convincingness", "specificity"
    """

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
        ["strongreject_rubric"],
        batch_size=batch_size
    )

    score_columns = {}
    score_columns["score"] = evaluator_results["score"]
    for subscore in ("refusal", "convincingness", "specificity"):
        if subscore in evaluator_results.column_names:
            score_columns[subscore] = evaluator_results[subscore]

    print(f"Generated {len(score_columns['score'])} scores with columns: {list(score_columns.keys())}")

    return score_columns


def main() -> None:
    # Parse arguments and setup
    args: argparse.Namespace = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")
    dir_path = os.path.join(args.results_dir, args.model_name, args.input_dir)

    # Ortho model subset, sweep of layers
    if args.subset:
        for layer in args.layers.split(","):
            layer = layer.strip()  # Remove any 
            if args.harmless:
                input_csv = f"ortho_output_subset_5_test_harmful_prompts_{args.type}_layer_{layer}_harmless.csv"
            else:
                input_csv = f"ortho_output_subset_5_test_harmful_prompts_{args.type}_layer_{layer}.csv"
            csv_path = os.path.join(dir_path, input_csv)

            # Check if input CSV exists
            if not os.path.exists(csv_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Warning: Input CSV not found at {csv_path}")
                print(f"  Skipping layer {layer} (ortho model may have produced no valid outputs)")
                continue

            print(f"Scoring dataset: {input_csv}")
        
            # Set default output CSV file
            base_name = os.path.splitext(input_csv)[0]
            output_csv = f"scored_{base_name}.csv"
            
            output_csv_path = os.path.join(
                args.results_dir, 
                args.model_name, 
                "attack_results", 
                output_csv
            )
            
            # Check if scored CSV already exists
            if os.path.exists(output_csv_path):
                print(f"Scored CSV already exists at {output_csv_path}")
                print("  Skipping (delete the file if you want to regenerate scores).")
                continue

            # Load data efficiently
            all_rows, fieldnames = load_data_efficiently(csv_path)
            
            # Compute scores
            scores = compute_scores(all_rows, args.batch_size)

            # Save original data with scores
            save_scored_csv(all_rows, scores, fieldnames, output_csv_path)
            print(f"Scoring complete! Use the filtering scripts with: --scored_csv {output_csv}")
    # For single CSV files
    else:
        csv_path = os.path.join(dir_path, args.input_csv)
        print(f"Scoring dataset: {args.input_csv}")

        # Check if input CSV exists
        if not os.path.exists(csv_path):
            print(f"Warning: Input CSV not found at {csv_path}")

        # Set default output CSV file
        base_name = os.path.splitext(args.input_csv)[0]
        output_csv = f"scored_{base_name}.csv"
        
        output_csv_path = os.path.join(args.results_dir, args.model_name, args.input_dir, output_csv
        )
        
        # Check if scored CSV already exists
        if os.path.exists(output_csv_path):
            print(f"Scored CSV already exists at {output_csv_path}")
            print("Delete the file if you want to regenerate scores.")
            return

        # Load data efficiently
        all_rows, fieldnames = load_data_efficiently(csv_path)
        
        # Compute scores
        scores = compute_scores(all_rows, args.batch_size)

        # Save original data with scores
        save_scored_csv(all_rows, scores, fieldnames, output_csv_path)
        print(f"Scoring complete! Use the filtering scripts with: --scored_csv {output_csv}")


if __name__ == "__main__":
    main()
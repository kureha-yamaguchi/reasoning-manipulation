"""
Script to filter Chain-of-Thought datasets based on pre-computed StrongReject evaluator scores.

Usage:
    python -m utils.filter_cot_datasets --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""

import csv
import os
import argparse
from typing import List, Dict, Tuple, Any

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter CoT datasets based on pre-computed StrongReject evaluator scores"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (for directory structure)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Results parent directory"
    )
    parser.add_argument(
        "--scored_csv",
        type=str,
        default="scored_all_harmful_prompts_cot5_out5.csv",
        help="CSV file containing pre-computed StrongReject scores"
    )
    parser.add_argument(
        "--lower_threshold",
        type=float,
        default=0.05,
        help="Lower threshold for creating refusal dataset"
    )
    parser.add_argument(
        "--upper_threshold",
        type=float,
        default=0.6,
        help="Upper threshold for creating non-refusal dataset"
    )

    return parser.parse_args()


def load_scored_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Load CSV data with pre-computed scores.
    
    Args:
        csv_path: Path to the CSV file containing scores
    
    Returns:
        List of dictionaries containing all CSV data including scores
    """
    print(f"Loading scored data from: {csv_path}")
    
    # First pass: count rows for progress bar
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for _ in reader)
    
    # Second pass: load data with progress bar
    all_rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_rows, desc="Loading scored data"):
            all_rows.append(row)
    
    print(f"Loaded {len(all_rows)} rows")
    return all_rows


def filter_cot_csv(
    scored_rows: List[Dict[str, str]],
    lower_threshold: float,
    upper_threshold: float
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Filter Chain-of-Thought responses based on StrongReject evaluator scores.
    
    This function processes evaluation scores in chunks (corresponding to multiple
    outputs from the same CoT), and classifies them as refusal or non-refusal
    based on whether all scores in a chunk meet the specified thresholds.
    
    Args:
        scored_rows: List of dictionaries containing CSV data with scores
        lower_threshold: Score threshold for refusal classification (all scores < threshold)
        upper_threshold: Score threshold for non-refusal classification (all scores > threshold)
    
    Returns:
        Tuple of (refusal_data, non_refusal_data)
    """
    
    # Initialize containers for filtered data
    refusal = []
    non_refusal = []

    chunk_size = 5  # Number of output variations per CoT

    # Process scores in chunks corresponding to same CoT
    for start in tqdm(range(0, len(scored_rows), chunk_size), desc="Processing CoT chunks"):
        end = min(start + chunk_size, len(scored_rows))
        same_cot_chunk = scored_rows[start:end]
        
        # Extract scores for this chunk
        chunk_scores = [float(row['strongreject_score']) for row in same_cot_chunk]
        
        if all(score < lower_threshold for score in chunk_scores):
            # Add to the refusal dataset - use the first row as representative
            refusal_row = same_cot_chunk[0].copy()
            refusal_row['output_scores'] = str(chunk_scores)
            refusal.append(refusal_row)
            
        elif all(score > upper_threshold for score in chunk_scores):
            # Add to the non-refusal dataset - use the first row as representative
            non_refusal_row = same_cot_chunk[0].copy()
            non_refusal_row['output_scores'] = str(chunk_scores)
            non_refusal.append(non_refusal_row)
    
    return refusal, non_refusal


def write_filtered_csv(filtered_data: List[Dict[str, str]], output_file: str, original_fieldnames: List[str]) -> None:
    """
    Write filtered data to a CSV file.
    
    Args:
        filtered_data: List of dictionaries containing filtered data
        output_file: Path to the output CSV file
        original_fieldnames: Original fieldnames from the scored CSV
    """
    if not filtered_data:
        print(f"No data to write to {output_file}")
        return
        
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine fieldnames - keep original ones and add output_scores if not present
    fieldnames = list(original_fieldnames)
    if 'output_scores' not in fieldnames:
        fieldnames.append('output_scores')
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in filtered_data:
            # Only write fields that are in fieldnames
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)
    
    print(f"Successfully exported {len(filtered_data)} rows to {output_file}")


def main() -> None:
    # Parse arguments
    args: argparse.Namespace = parse_args()
    
    scored_csv_path = os.path.join(args.results_dir, args.model_name, "dataset", args.scored_csv)
    
    # Check if scored CSV file exists
    if not os.path.exists(scored_csv_path):
        print(f"Error: Scored CSV file not found at {scored_csv_path}")
        print("Run the scoring script first to generate scores.")
        return

    # Load pre-computed scores
    scored_rows = load_scored_csv(scored_csv_path)
    
    # Get original fieldnames
    with open(scored_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames

    # Filter based on evaluation scores
    print(f"Total samples: {len(scored_rows)}")
    refusal, non_refusal = filter_cot_csv(
        scored_rows,
        args.lower_threshold,
        args.upper_threshold
    )
    
    print(f"Refusal samples: {len(refusal)}")
    print(f"Non-refusal samples: {len(non_refusal)}")

    # Save refusal dataset
    output_refusal_path = os.path.join(
        args.results_dir,
        args.model_name,
        "dataset",
        f"refusal_{args.lower_threshold}.csv"
    )
    write_filtered_csv(refusal, output_refusal_path, original_fieldnames)

    # Save non-refusal dataset
    output_nonrefusal_path = os.path.join(
        args.results_dir,
        args.model_name,
        "dataset",
        f"nonrefusal_{args.upper_threshold}.csv"
    )
    write_filtered_csv(non_refusal, output_nonrefusal_path, original_fieldnames)


if __name__ == "__main__":
    main()
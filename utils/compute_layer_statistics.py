"""
Script to compute mean and standard deviation of StrongReject scores across multiple layers for a subset of the holdout test dataset.

This script reads scored CSV files for different layers and computes statistics on the StrongReject scores for each layer.

Input Files:
    Reads scored CSV files from:
    results/{model_name}/dataset/scored_ortho_output_subset_5_test_harmful_prompts_{type}_layer_{layer}.csv
    
    Where {layer} comes from --layer argument (comma-separated list like "16,17,18,19")

Output:
    Prints mean and standard deviation of StrongReject scores for each layer.
    Saves summary statistics to JSON file in results/{model_name}/attack_results/.

Usage:
    uv run -m utils.compute_layer_statistics --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --type cot --layer 16,17,18,19
"""

import argparse
import os
import sys
import csv
import json
from typing import List, Dict, Tuple
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute StrongReject score statistics across multiple layers"
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
        "--type",
        type=str,
        required=True,
        help="Type of data (cot or baseline)"
    )
    parser.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Comma-separated list of layers (e.g., '16,17,18,19')"
    )
    
    return parser.parse_args()


def load_scores_from_csv(csv_path: str) -> List[float]:
    """
    Load StrongReject scores from a CSV file.
    
    Args:
        csv_path: Path to the scored CSV file
    
    Returns:
        List of StrongReject scores as floats
    """
    scores = []
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return scores
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check if strongreject_score column exists
            if 'strongreject_score' not in reader.fieldnames:
                print(f"Warning: 'strongreject_score' column not found in {csv_path}")
                print(f"Available columns: {reader.fieldnames}")
                return scores
            
            for row in reader:
                score_str = row.get('strongreject_score', '').strip()
                if score_str:
                    try:
                        score = float(score_str)
                        scores.append(score)
                    except ValueError:
                        print(f"Warning: Could not convert '{score_str}' to float, skipping")
                        continue
        
        return scores
    
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return scores


def compute_statistics(scores: List[float]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of scores.
    
    Args:
        scores: List of scores
    
    Returns:
        Tuple of (mean, std_dev)
    """
    if not scores:
        return float('nan'), float('nan')
    
    if len(scores) == 1:
        return scores[0], 0.0
    
    mean = statistics.mean(scores)
    std_dev = statistics.stdev(scores)  # Sample standard deviation
    
    return mean, std_dev


def main() -> None:
    # Parse arguments
    args: argparse.Namespace = parse_args()
    
    # Parse layers
    layers = [layer.strip() for layer in args.layer.split(',')]
    
    if not layers:
        print("Error: No layers specified")
        sys.exit(1)
    
    print(f"{'='*60}")
    print(f"Computing StrongReject Score Statistics")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Type: {args.type}")
    print(f"Layers: {', '.join(layers)}")
    print(f"{'='*60}\n")
    
    # Base directory for CSV files
    dataset_dir = os.path.join(args.results_dir, args.model_name, "dataset")
    
    # Store results for each layer
    results: Dict[str, Dict[str, float]] = {}
    
    # Process each layer
    for layer in layers:
        # Construct CSV filename
        scored_csv = f"scored_ortho_output_subset_5_test_harmful_prompts_{args.type}_layer_{layer}.csv"
        csv_path = os.path.join(dataset_dir, scored_csv)
        
        print(f"Processing layer {layer}...")
        print(f"  File: {scored_csv}")
        
        # Load scores
        scores = load_scores_from_csv(csv_path)
        
        if not scores:
            print(f"  Warning: No scores found for layer {layer}")
            results[layer] = {
                'mean': float('nan'),
                'std_dev': float('nan'),
                'count': 0
            }
            continue
        
        # Compute statistics
        mean, std_dev = compute_statistics(scores)
        
        results[layer] = {
            'mean': mean,
            'std_dev': std_dev,
            'count': len(scores)
        }
        
        print(f"  Found {len(scores)} scores")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std Dev: {std_dev:.6f}")
        print()
    
    # Print summary table
    print(f"{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<10} {'Count':<10} {'Mean':<15} {'Std Dev':<15}")
    print(f"{'-'*60}")
    
    for layer in layers:
        result = results[layer]
        count = result['count']
        mean = result['mean']
        std_dev = result['std_dev']
        
        if count > 0:
            print(f"{layer:<10} {count:<10} {mean:<15.6f} {std_dev:<15.6f}")
        else:
            print(f"{layer:<10} {count:<10} {'N/A':<15} {'N/A':<15}")
    
    print(f"{'='*60}")
    
    # Save results to JSON file
    output_dir = os.path.join(args.results_dir, args.model_name, "attack_results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"layer_statistics_{args.type}_layers_{args.layer.replace(',', '_')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Check if all layers were processed successfully
    failed_layers = [layer for layer in layers if results[layer]['count'] == 0]
    if failed_layers:
        print(f"\nWarning: Failed to process layers: {', '.join(failed_layers)}")
        sys.exit(1)
    else:
        print("\nAll layers processed successfully!")


if __name__ == "__main__":
    main()


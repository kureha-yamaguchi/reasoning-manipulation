"""
Script to score model output generations using rubric based StrongReject evaluator.

Example usage:
    CUDA_VISIBLE_DEVICES=0 uv run -m utils.heuristic.compute_scores_rollouts \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model generations using StrongReject evaluator")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size into strongreject evaluator")
    parser.add_argument("--repetitions", type=int, default=10,
                        help="Number of output variations per prompt")
    parser.add_argument("--quadrant", action="store_true",
                        help="Scoring quadrant rollouts?")
    return parser.parse_args()


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
    # Pre-populate score columns so the schema is consistent across all
    # multiprocessing workers (avoids KeyError when some batches don't
    # return every subscore).
    n = len(prompts)
    output_dataset: Dataset = Dataset.from_dict({
        "forbidden_prompt": prompts,
        "response": outputs,
        "score": [float('nan')] * n,
        "refusal": [float('nan')] * n,
        "convincingness": [float('nan')] * n,
        "specificity": [float('nan')] * n,
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

def save_scored(input_paths, args):
    for input_path in input_paths:
        output_path = os.path.join(
            os.path.dirname(input_path),
            f'scored_{os.path.basename(input_path)}'
        )
        if os.path.exists(output_path):
            print(f"\nSkipping {input_path} — scored file already exists: {output_path}")
            continue
            
        # Load data
        print(f"\nReading data from: {input_path}")

        # Build the list of score columns to add
        score_field_map = {
            "score": "strongreject_score",
            "refusal": "strongreject_refusal",
            "convincingness": "strongreject_convincingness",
            "specificity": "strongreject_specificity",
        }

        all_rows = []
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in tqdm(reader, desc="Loading data"):
                all_rows.append(row)

        print(f"Loaded {len(all_rows)} rows")

        # Compute scores
        score_columns = compute_scores(all_rows, args.batch_size)

        # Save with scores
        new_fieldnames = list(fieldnames)

        for key, csv_col in score_field_map.items():
            if key in score_columns and csv_col not in new_fieldnames:
                new_fieldnames.append(csv_col)

        # Write CSV with scores
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
            writer.writeheader()

            for i, row in enumerate(tqdm(all_rows, desc="Writing scored CSV")):
                scored_row = row.copy()
                for key, csv_col in score_field_map.items():
                    if key in score_columns:
                        scored_row[csv_col] = score_columns[key][i]
                writer.writerow(scored_row)

        print(f"Saved scored results to: {output_path}")

def main() -> None:
    args = parse_args()

    if args.quadrant:
        input_csv = f'full_resample_prompt*_cot*_rep_{args.repetitions}.csv'
        # Score Refusal
        print("=====Scoring refusal=====")
        pattern_quadrant = os.path.join(
            args.results_dir,
            args.model_name,
            'dataset',
            'quadrants',
            input_csv
        )
        input_paths_quadrant = sorted(glob.glob(pattern_quadrant))
        if not input_paths_quadrant:
            print(f"No input files found matching pattern: {pattern_quadrant}")
            return
        print(f"Found {len(input_paths_quadrant)} input file(s): {[os.path.basename(p) for p in input_paths_quadrant]}")
        save_scored(input_paths_quadrant, args)
    else:
        input_csv = f'full_resample_*_rep_{args.repetitions}.csv'

        # Score Refusal
        print("=====Scoring refusal=====")
        pattern_refusal = os.path.join(
            args.results_dir,
            args.model_name,
            'dataset',
            'refusal',
            input_csv
        )
        input_paths_refusal = sorted(glob.glob(pattern_refusal))
        if not input_paths_refusal:
            print(f"No input files found matching pattern: {pattern_refusal}")
            return
        print(f"Found {len(input_paths_refusal)} input file(s): {[os.path.basename(p) for p in input_paths_refusal]}")
        save_scored(input_paths_refusal, args)

        # Score Nonrefusal
        print("=====Scoring nonrefusal=====")
        pattern_nonrefusal = os.path.join(
            args.results_dir,
            args.model_name,
            'dataset',
            'nonrefusal',
            input_csv
        )
        input_paths_nonrefusal = sorted(glob.glob(pattern_nonrefusal))
        if not input_paths_nonrefusal:
            print(f"No input files found matching pattern: {pattern_nonrefusal}")
            return
        print(f"Found {len(input_paths_nonrefusal)} input file(s): {[os.path.basename(p) for p in input_paths_nonrefusal]}")
        save_scored(input_paths_nonrefusal, args)

if __name__ == "__main__":
    main()
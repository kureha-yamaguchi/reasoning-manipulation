"""
Script to score model output generations using StrongReject evaluator.

Example usage:
    CUDA_VISIBLE_DEVICES=0 uv run -m utils.heuristic.compute_scores_transfer \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --index_number 3 \
        --cot_number 1
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model generations using StrongReject evaluator")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--index_number", type=int, required=True,
                        help="Index number as per quadrant_output.txt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size into strongreject evaluator"
    )
    return parser.parse_args()

def compute_scores(all_rows: List[Dict[str, str]], batch_size: int) -> List[float]:
    """
    Evaluate output scores using StrongReject evaluator.
    
    Args:
        all_rows: List of rows from the CSV file

    Returns:
        List of StrongReject scores
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
        ["strongreject_finetuned"],
        batch_size=batch_size
    )
    
    scores = evaluator_results["score"]
    print(f"Generated {len(scores)} scores")

    return scores

def main() -> None:
    args = parse_args()

    for i in range(5):
        input_csv = f'resampling_results_idx{args.index_number}_cot{i+1}_transfer.csv'
        input_path = f"results/{args.model_name}/dataset/first_sentence/{input_csv}"
        base_name = os.path.splitext(input_csv)[0]
        output_path = f"results/{args.model_name}/dataset/first_sentence/scored_{base_name}.csv"

        # Load data
        print(f"Reading data from: {input_path}")
        all_rows = []
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in tqdm(reader, desc="Loading data"):
                all_rows.append(row)
        
        print(f"Loaded {len(all_rows)} rows")

        # Compute scores
        scores = compute_scores(all_rows, args.batch_size)

        # Save with scores
        new_fieldnames = list(fieldnames) + ['strongreject_score']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            for i, row in enumerate(tqdm(all_rows, desc="Writing scored CSV")):
                row['strongreject_score'] = scores[i]
                writer.writerow(row)

        print(f"Saved scored results to: {output_path}")


if __name__ == "__main__":
    main()

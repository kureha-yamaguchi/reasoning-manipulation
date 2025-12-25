"""
Script to score model output generations using StrongReject evaluator.

Example usage:
    CUDA_VISIBLE_DEVICES=0 uv run -m utils.heuristic.compute_scores \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --input_csv train_harmful_prompts_out25.csv
"""

import argparse
import csv
import os

from tqdm import tqdm
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model generations using StrongReject evaluator")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV filename (e.g. train_harmful_prompts_out25.csv)")
    return parser.parse_args()

def compute_scores(all_rows: List[Dict[str, str]]) -> List[float]:
    """
    Evaluate output scores using StrongReject evaluator.
    
    Args:
        all_rows: List of rows from the CSV file

    Returns:
        List of StrongReject scores
    """
    EVALUATOR = "strongreject_rubric"

    # Extract prompts and outputs for evaluation
    prompts = [row["prompt"] for row in all_rows]
    outputs = [row["output"] for row in all_rows]

    # Create output dataset for evaluation
    output_dataset: Dataset = Dataset.from_dict({
        "forbidden_prompt": prompts,
        "response": outputs
    })

    # run the rubric fully locally using vLLM (no API keys needed)
    local_eval_dataset = evaluate_dataset(
        output_dataset,
        [EVALUATOR],
        use_local_vllm=True,
        vllm_model="google/gemma-3-27b-it",
    )
    
    scores = evaluator_results["score"]
    print(f"Generated {len(scores)} scores")

    return scores

def main() -> None:
    args = parse_args()
    
    input_path = f"results/{args.model_name}/dataset/{args.input_csv}"
    base_name = os.path.splitext(args.input_csv)[0]
    output_path = f"results/{args.model_name}/dataset/scored_{base_name}.csv"

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
    scores = compute_scores(all_rows)

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

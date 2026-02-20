"""
Script to score model output generations using StrongReject evaluator.

Example usage:

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.heuristic.compute_scores_transfer \
  --from_model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --to_model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --index_number 3

"""

import argparse
import csv
import glob
import os
from typing import Dict, List
from tqdm import tqdm
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model generations using StrongReject evaluator")
    parser.add_argument("--from_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="First CoT sentence taken from this model")
    parser.add_argument("--to_model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Prefill attack applied on this model")
    parser.add_argument("--index_number", type=int, required=True,
                        help="Index number as per quadrant_output.txt")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size into strongreject evaluator")
    return parser.parse_args()

def compute_scores(all_rows: List[Dict[str, str]], batch_size: int) -> List[float]:
    """
    Evaluate output scores using StrongReject evaluator.

    Args:
        all_rows: List of rows from the CSV file

    Returns:
        List of StrongReject scores
    """
    prompts = [row["prompt"] for row in all_rows]
    outputs = [row["output"] for row in all_rows]

    output_dataset: Dataset = Dataset.from_dict({
        "forbidden_prompt": prompts,
        "response": outputs
    })

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
    from_model_short = args.from_model_name.split("/")[-1]
    resample_dir = f"results/{args.to_model_name}/dataset/resample"
    pattern = os.path.join(
        resample_dir,
        f'transfer_results_idx{args.index_number}_cot*_transfer_from_{from_model_short}.csv'
    )

    input_paths = sorted(glob.glob(pattern))
    if not input_paths:
        print(f"No input files found matching pattern: {pattern}")
        return

    print(f"Found {len(input_paths)} input file(s): {[os.path.basename(p) for p in input_paths]}")

    for input_path in input_paths:
        input_csv = os.path.basename(input_path)
        base_name = os.path.splitext(input_csv)[0]
        output_path = os.path.join(resample_dir, f"scored_{base_name}.csv")

        # Load data
        print(f"\nReading data from: {input_path}")
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
            for idx, row in enumerate(tqdm(all_rows, desc="Writing scored CSV")):
                row['strongreject_score'] = scores[idx]
                writer.writerow(row)

        print(f"Saved scored results to: {output_path}")


if __name__ == "__main__":
    main()
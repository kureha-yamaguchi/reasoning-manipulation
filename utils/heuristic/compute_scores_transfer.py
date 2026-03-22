"""
Script to score model output generations using rubric based StrongReject evaluator.

Example usage:

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run -m utils.heuristic.compute_scores_transfer \
  --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --transfer_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --transfer_model_2 deepseek/deepseek-reasoner \
  --prompt_index 174

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
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="First CoT sentence taken from this model")
    parser.add_argument("--transfer_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Prefill attack applied on this model")
    parser.add_argument("--transfer_model_2", type=str, default=None,
                        help="Prefill attack also applied on this model")
    parser.add_argument("--prompt_index", type=int, required=True,
                        help="Original prompt index")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size into strongreject evaluator")
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

def save_scored(model_name, input_paths, args):
    for input_path in input_paths:
        input_csv = os.path.basename(input_path)
        base_name = os.path.splitext(input_csv)[0]
        output_path = os.path.join(
            args.results_dir,
            model_name,
            'dataset',
            'resample',
            f"scored_{base_name}.csv")
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

    base_model_short = args.base_model.split("/")[-1]

    base_csv = f'resampling_results_idx{args.prompt_index}_cot*.csv'
    transfer_csv = f'transfer_results_idx{args.prompt_index}_cot*_transfer_from_{base_model_short}.csv'

    pattern_base = os.path.join(
        args.results_dir,
        args.base_model,
        'dataset',
        'resample',
        base_csv
    )
    pattern_transfer = os.path.join(
        args.results_dir,
        args.transfer_model,
        'dataset',
        'resample',
        transfer_csv
    )

    input_paths_base = sorted(glob.glob(pattern_base))
    input_paths_transfer = sorted(glob.glob(pattern_transfer))

    if not input_paths_base:
        print(f"No input files found matching pattern: {pattern_base}")
        return
    if not input_paths_transfer:
        print(f"No input files found matching pattern: {pattern_transfer}")
        return

    print(f"Found {len(input_paths_base)} input file(s): {[os.path.basename(p) for p in input_paths_base]}")
    print(f"Found {len(input_paths_transfer)} input file(s): {[os.path.basename(p) for p in input_paths_transfer]}")
    
    save_scored(args.base_model, input_paths_base, args)
    save_scored(args.transfer_model, input_paths_transfer, args)

    # if args.transfer_model_2 is not None:
    #     pattern_transfer2 = os.path.join(
    #         args.results_dir,
    #         args.transfer_model_2,
    #         'dataset',
    #         'resample',
    #         transfer_csv
    #     )
    #     input_paths_transfer2 = sorted(glob.glob(pattern_transfer2))
    #     if not input_paths_transfer2:
    #         print(f"No input files found matching pattern: {pattern_transfer2}")
    #         return
    #     print(f"Found {len(input_paths_transfer2)} input file(s): {[os.path.basename(p) for p in input_paths_transfer2]}")
    #     save_scored(args.transfer_model_2, input_paths_transfer2, args)


if __name__ == "__main__":
    main()
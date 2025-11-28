"""
Filter rows from all_harmful_prompts_cot5_out5.csv to keep only prompts that exist in test_harmful_prompts.csv. Saves me having to re-run the batch_generation_cot_output.py script on the test dataset, given I already had the generations for all_harmful_prompts.

Example:
    uv run -m utils.retrospective.retrospective_test_gen --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter all_harmful_prompts_cot5_out5.csv to keep only test prompts"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-8B)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load test harmful prompts
    test_prompts_path = Path("dataset/test_harmful_prompts.csv")
    test_prompts = pd.read_csv(test_prompts_path)
    # Convert from pandas series to python set for faster lookup
    test_prompts_set = set(test_prompts["prompt"])
    print(f"Loaded {len(test_prompts_set)} test prompts from {test_prompts_path}")

    # Define paths
    dataset_dir = Path("results") / args.model_name / "dataset"
    input_csv = dataset_dir / "all_harmful_prompts_cot5_out5.csv"
    output_csv = dataset_dir / "test_harmful_prompts_cot5_out5.csv"

    # Check if input file exists
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    # Load the full dataset
    print(f"Loading dataset from {input_csv}...")
    full_df = pd.read_csv(input_csv)
    print(f"Full dataset rows: {len(full_df)}")

    # Filter rows where prompt is in test_prompts_set
    filtered_df = full_df[full_df["prompt"].isin(test_prompts_set)]
    print(f"Filtered dataset rows: {len(filtered_df)}")

    # Save the filtered dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved filtered dataset to {output_csv}")


if __name__ == "__main__":
    main()


"""
Filter datasets retrospectively, removing prompts in the holdout set from the refusal/nonrefusal training set, keeping only prompts that exist in train_harmful_prompts.csv

Usage:
    uv run -m utils.retrospective.retrospective_filter --model_name MODEL --type TYPE

Example:
    uv run -m utils.retrospective.retrospective_filter --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --type cot
"""
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Filter datasets retrospectively, removing prompts in the holdout set from the training set")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B', 
                        help='Model name')
    parser.add_argument('--type', type=str, default='baseline', 
                        help="CoT tokens (cot) or 3 tokens at the end of prompt (baseline) or whole prompt (prompt)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load train harmful prompts
    train_prompts_path = Path("dataset/train_harmful_prompts.csv")
    train_prompts = pd.read_csv(train_prompts_path)
    # Convert from pandas series to python set for faster lookup
    train_prompts_set = set(train_prompts["prompt"])

    # Define paths
    holdout_dir = Path("results") / args.model_name / "dataset"
    dataset_dir = holdout_dir / "legacy"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    # Process refusal file
    refusal_input = dataset_dir / f"refusal_0.05_{args.type}.csv"
    refusal_df = pd.read_csv(refusal_input)
    print(f"refusal_df rows: {len(refusal_df)}")
    refusal_filtered = refusal_df[refusal_df["prompt"].isin(train_prompts_set)]
    print(f"refusal_filtered rows: {len(refusal_filtered)}")
    refusal_output = holdout_dir / f"refusal_0.05_{args.type}.csv"
    refusal_filtered.to_csv(refusal_output, index=False)

    # Process nonrefusal file
    nonrefusal_input = dataset_dir / f"nonrefusal_0.6_{args.type}.csv"
    nonrefusal_df = pd.read_csv(nonrefusal_input)
    print(f"nonrefusal_df rows: {len(nonrefusal_df)}")
    nonrefusal_filtered = nonrefusal_df[nonrefusal_df["prompt"].isin(train_prompts_set)]
    print(f"nonrefusal_filtered rows: {len(nonrefusal_filtered)}")
    nonrefusal_output = holdout_dir / f"nonrefusal_0.6_{args.type}.csv"
    nonrefusal_filtered.to_csv(nonrefusal_output, index=False)


if __name__ == "__main__":
    main()

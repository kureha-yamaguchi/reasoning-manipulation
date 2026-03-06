"""
Simple script to pull a dataset from Hugging Face Hub and save as CSV.
uv run -m utils.csv_from_hf \
  --hf_username kureha295 \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --dataset train_harmful_prompts_cot5_out5
"""

from datasets import load_dataset
import argparse
import os
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pull a dataset from Hugging Face Hub and save as CSV")
    parser.add_argument("--hf_username", type=str, default="kureha295",
                       help="Huggingface username")
    parser.add_argument("--model_name", type=str,
                       help="Model name (used for repo name and local path)")
    parser.add_argument("--dataset", type=str, default="train_harmful_prompts_cot5_out5",
                       help="Dataset name (without .csv extension)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to download (default: train)")
    return parser.parse_args()


def main():
    args = parse_args()

    model_short = args.model_name.split("/")[-1]
    dataset_short = Path(args.dataset).stem

    repo_name = f"{args.hf_username}/{model_short}_{dataset_short}"

    # Download dataset from HF
    print(f"Downloading dataset from: https://huggingface.co/datasets/{repo_name}")
    ds = load_dataset(repo_name, split=args.split)
    df = ds.to_pandas()

    # Save to local CSV
    out_dir = os.path.join('results', args.model_name, 'dataset')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_short}.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

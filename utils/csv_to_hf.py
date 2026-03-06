"""
Simple script to push a dataset to Hugging Face Hub.
uv run -m utils.csv_to_hf \
  --hf_username kureha295 \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --dataset scored_train_harmful_prompts_cot5_out5.csv
"""

from datasets import Dataset, DatasetDict
from huggingface_hub import login
import argparse
import pandas as pd
import os
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create base dataset of harmful prompts")
    parser.add_argument("--hf_username", type=str, default="kureha295",
                       help="Huggingface username)")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                       help="Huggingface username)")       
    parser.add_argument("--dataset", type=str, default="train_harmful_prompts_cot5_out5.csv",
                       help="Dataset to be pushed")
    return parser.parse_args()

def push_dataset_to_hub(
    data: dict,
    repo_name: str,
    private: bool = False
):
    """
    Push a dataset to Hugging Face Hub.
    
    Args:
        data: Dictionary with column names as keys and lists of values as values
        repo_name: Repository name (e.g., "username/dataset-name")
        private: Whether the dataset should be private
        token: Hugging Face API token (optional if already logged in)
    """
    
    # Create dataset from dictionary
    dataset = Dataset.from_dict(data)
    
    # Push to Hub
    dataset.push_to_hub(
        repo_id=repo_name,
        private=private
    )
    
    print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")


def push_dataset_with_splits(
    train_data: dict,
    test_data: dict = None,
    repo_name: str = None,
    private: bool = False
):
    """
    Push a dataset with train/test splits to Hugging Face Hub.
    
    Args:
        train_data: Training data dictionary
        test_data: Test data dictionary (optional)
        repo_name: Repository name (e.g., "username/dataset-name")
        private: Whether the dataset should be private
        token: Hugging Face API token (optional if already logged in)
    """
    
    # Create dataset dictionary with splits
    splits = {"train": Dataset.from_dict(train_data)}
    
    if test_data:
        splits["test"] = Dataset.from_dict(test_data)
    
    dataset_dict = DatasetDict(splits)
    
    # Push to Hub
    dataset_dict.push_to_hub(
        repo_id=repo_name,
        private=private
    )
    
    print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")


def main():
    args = parse_args()
    
    CSV_PATH = os.path.join('results', args.model_name, 'dataset', args.dataset)
    
    model_short = args.model_name.split("/")[-1]
    dataset_short = Path(args.dataset).stem  # removes any extension

    REPO_NAME = os.path.join(args.hf_username, f"{model_short}_{dataset_short}")
    
    # Load CSV into a pandas DataFrame, then convert to dict
    df = pd.read_csv(CSV_PATH)
    data = df.to_dict(orient="list")

    # Push the dataset
    push_dataset_to_hub(
        data=data,
        repo_name=REPO_NAME,
        private=False  # Set to True for private datasets
    )
    

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script to combine test and train CSV files from legacy directory
into combined dataset files for model results.

Usage:
    uv run -m utils.retrospective.combine_datasets --model-name Qwen/Qwen3-8B
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Combine test and train CSV files from legacy directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (e.g., 'Qwen/Qwen3-8B')"
    )
    args = parser.parse_args()
    
    # Base directory - script is in utils/retrospective, need to point to results/{model_name}
    base_dir = Path("results") / args.model_name
    legacy_dir = base_dir / "dataset" / "legacy"
    output_dir = base_dir / "dataset"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the file combinations
    combinations = [
        {
            "test": "test_nonrefusal_0.6_baseline.csv",
            "train": "train_nonrefusal_0.6_baseline.csv",
            "output": "nonrefusal_0.6_baseline.csv"
        },
        {
            "test": "test_nonrefusal_0.6_cot.csv",
            "train": "train_nonrefusal_0.6_cot.csv",
            "output": "nonrefusal_0.6_cot.csv"
        },
        {
            "test": "test_refusal_0.05_baseline.csv",
            "train": "train_refusal_0.05_baseline.csv",
            "output": "refusal_0.05_baseline.csv"
        },
        {
            "test": "test_refusal_0.05_cot.csv",
            "train": "train_refusal_0.05_cot.csv",
            "output": "refusal_0.05_cot.csv"
        }
    ]
    
    # Process each combination
    for combo in combinations:
        test_file = legacy_dir / combo["test"]
        train_file = legacy_dir / combo["train"]
        output_file = output_dir / combo["output"]
        
        print(f"\nCombining {combo['test']} and {combo['train']}...")
        
        # Check if files exist
        if not test_file.exists():
            print(f"  ⚠️  Warning: {test_file} does not exist, skipping...")
            continue
        if not train_file.exists():
            print(f"  ⚠️  Warning: {train_file} does not exist, skipping...")
            continue
        
        # Read the CSV files
        test_df = pd.read_csv(test_file)
        train_df = pd.read_csv(train_file)
        
        # Combine the dataframes
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Save the combined dataframe
        combined_df.to_csv(output_file, index=False)
        
        print(f"  ✓ Created {combo['output']}")
        print(f"    - Train rows: {len(train_df)}")
        print(f"    - Test rows: {len(test_df)}")
        print(f"    - Total rows: {len(combined_df)}")
    
    print("\n✓ All files combined successfully!")


if __name__ == "__main__":
    main()


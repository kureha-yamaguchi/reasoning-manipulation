"""
Reads dataset/all_harmful_prompts.csv and creates a train-test split.
Takes 75% of data for training set and 25% for test set.
Shuffles the dataset before splitting to ensure randomization.

Usage:
python -m utils.create_holdout_set --train_set_split 0.75
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create train-test split from all_harmful_prompts.csv"
    )
    parser.add_argument("--input_file", type=str, default="dataset/all_harmful_prompts.csv",
                        help="Path to input CSV file")
    parser.add_argument("--train_set_split", type=float, default=0.75,
                        help="Train split proportion between 0-1 (default: 0.75)")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Directory to save output files")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def create_train_test_split(input_path, train_set_split, random_state=42):
    """
    Create train-test split from a single dataset.

    Parameters:
    -----------
    input_path : str
        Path to the input CSV file
    train_set_split: float
        Train split proportion between 0 and 1
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    dict : Dictionary containing the splits and metadata
    """

    # Read the CSV file
    print(f"Reading CSV file: {input_path}...")
    df = pd.read_csv(input_path)

    # Get the number of rows in the dataset
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # Calculate train and test sizes
    train_size = int(train_set_split * total_rows)
    test_size = total_rows - train_size

    print(f"Train set size: {train_size} ({train_set_split:.0%})")
    print(f"Test set size: {test_size} ({1-train_set_split:.0%})")

    # Shuffle the dataset
    print("\nShuffling dataset...")
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Create train-test split
    print("Creating train-test split...")
    train_df = df_shuffled.head(train_size)
    test_df = df_shuffled.tail(test_size)

    # Verify splits
    print("\nVerifying splits:")
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print(f"Total: {len(train_df) + len(test_df)} rows")

    # Return the splits
    return {
        'total_rows': total_rows,
        'train_size': train_size,
        'test_size': test_size,
        'train_df': train_df,
        'test_df': test_df
    }


def main():
    """
    Main function to execute the train-test split creation.
    """
    # Parse arguments
    args = parse_args()

    try:
        # Create the train-test split
        results = create_train_test_split(
            input_path=args.input_file,
            train_set_split=args.train_set_split,
            random_state=args.random_state
        )

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Save the splits to CSV files
        print("\nSaving splits to CSV files...")
        train_output_path = os.path.join(args.output_dir, 'train_harmful_prompts.csv')
        test_output_path = os.path.join(args.output_dir, 'test_harmful_prompts.csv')

        results['train_df'].to_csv(train_output_path, index=False)
        results['test_df'].to_csv(test_output_path, index=False)

        print(f"Train set saved to: {train_output_path}")
        print(f"Test set saved to: {test_output_path}")

        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print("Successfully created train-test split")
        print(f"Input file: {args.input_file}")
        print(f"Total rows: {results['total_rows']}")
        print(f"Train set size ({args.train_set_split:.0%}): {results['train_size']}")
        print(f"Test set size ({1-args.train_set_split:.0%}): {results['test_size']}")
        print(f"Random seed: {args.random_state}")

    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        print(f"Please ensure the file exists at: {args.input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

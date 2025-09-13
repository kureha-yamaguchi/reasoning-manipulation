"""
Reads both refusal and non-refusal files, finds the smaller dataset size and stores it as variable n,
randomizes both datasets by shuffling rows independently for both refusal and non-refusal datasets

Takes args.train_set_split% of n rows for training sets
Takes (1-args.train_set_split)% of n rows for test sets
Ensures both refusal and non-refusal have the same number of samples

Usage:
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m utils.create_train_test_split \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --train_set_split 0.75
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-stage CoT generation: Generate CoT responses, then generate outputs for each CoT"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
                        help="Model to use for generation")
    parser.add_argument("--refusal_dataset", type=str, default="refusal_0.05.csv", 
                        help="Name of the refusal dataset")
    parser.add_argument("--nonrefusal_dataset", type=str, default="nonrefusal_0.6.csv", 
                        help="Name of the non-refusal dataset")
    parser.add_argument("--train_set_split", type=float, default="0.75", 
                        help="Train split proportion between 0-1")
    return parser.parse_args()


def create_balanced_splits(refusal_path, non_refusal_path, train_set_split, random_state=42):
    """
    Create balanced train-test splits from refusal and non-refusal datasets.
    
    Parameters:
    -----------
    refusal_path : str
        Path to the refusal CSV file
    non_refusal_path : str
        Path to the non-refusal CSV file
    train_set_split: float
        Train split proportion between 0 and 1
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary containing the splits and metadata
    """
    
    # Read the CSV files
    print("Reading CSV files...")
    refusal_df = pd.read_csv(refusal_path)
    non_refusal_df = pd.read_csv(non_refusal_path)
    
    # Get the number of rows in each dataset
    refusal_rows = len(refusal_df)
    non_refusal_rows = len(non_refusal_df)
    
    print(f"Refusal dataset rows: {refusal_rows}")
    print(f"Non-refusal dataset rows: {non_refusal_rows}")
    
    # Determine n (smaller dataset size)
    n = min(refusal_rows, non_refusal_rows)
    print(f"Using n = {n} (size of smaller dataset)")
    
    # Calculate train and test sizes
    train_size = int(train_set_split * n)
    test_size = n - train_size
    
    print(f"Train set size: {train_size}")
    print(f"Test set size: {test_size}")
    
    # Shuffle the datasets using Pandas method that randomly samples from 100% of the rows from the dataframe
    print("\nShuffling datasets...")
    refusal_df_shuffled = refusal_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    non_refusal_df_shuffled = non_refusal_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Take only n samples from each dataset (in case one is larger)
    refusal_df_balanced = refusal_df_shuffled.head(n)
    non_refusal_df_balanced = non_refusal_df_shuffled.head(n)
    
    # Create train-test splits
    print("Creating train-test splits...")
    
    # Split refusal dataset
    refusal_train = refusal_df_balanced.head(train_size)
    refusal_test = refusal_df_balanced.iloc[train_size:train_size + test_size]
    
    # Split non-refusal dataset
    non_refusal_train = non_refusal_df_balanced.head(train_size)
    non_refusal_test = non_refusal_df_balanced.iloc[train_size:train_size + test_size]
    
    # Verify splits
    print("\nVerifying splits:")
    print(f"Refusal train: {len(refusal_train)} rows")
    print(f"Refusal test: {len(refusal_test)} rows")
    print(f"Non-refusal train: {len(non_refusal_train)} rows")
    print(f"Non-refusal test: {len(non_refusal_test)} rows")
    
    # Return the splits
    return {
        'n': n,
        'train_size': train_size,
        'test_size': test_size,
        'refusal_train': refusal_train,
        'refusal_test': refusal_test,
        'non_refusal_train': non_refusal_train,
        'non_refusal_test': non_refusal_test
    }


def main():
    """
    Main function to execute the balanced split creation.
    """
    # Configure file paths
    args = parse_args()
    refusal_csv_path = os.path.join('results', args.model_name, 'dataset', args.refusal_dataset) 
    non_refusal_csv_path = os.path.join('results', args.model_name, 'dataset', args.nonrefusal_dataset) 
    
    try:
        # Create the balanced splits
        results = create_balanced_splits(
            refusal_path=refusal_csv_path,
            non_refusal_path=non_refusal_csv_path,
            train_set_split=args.train_set_split,
            random_state=42  # Set seed for reproducibility
        )

        # Save the splits to CSV files
        print("\nSaving splits to CSV files...")
        save_dir = os.path.join('results', args.model_name, 'dataset') 
        results['refusal_train'].to_csv(os.path.join(save_dir,'refusal_train.csv'), index=False)
        results['refusal_test'].to_csv(os.path.join(save_dir,'refusal_test.csv'), index=False)
        results['non_refusal_train'].to_csv(os.path.join(save_dir,'non_refusal_train.csv'), index=False)
        results['non_refusal_test'].to_csv(os.path.join(save_dir,'non_refusal_test.csv'), index=False)

        os.path.join('results', args.model_name, 'dataset', args.refusal_dataset) 
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Successfully created balanced train-test splits")
        print(f"Base size (n): {results['n']}")
        print(f"Train set size ({args.train_set_split:.0%}): {results['train_size']}")
        test_set_split=1-args.train_set_split
        print(f"Test set size ({test_set_split:.0%}): {results['test_size']}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        print("Please update the file paths in the main() function")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
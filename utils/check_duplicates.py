"""
Script to check for duplicate prompts across multiple CSV files.
Handles prompts containing quotation marks properly.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m utils.check_duplicates
def read_prompts_from_csv(filepath):
    """
    Read prompts from a CSV file with a 'prompt' column.
    Args:
        filepath: Path to the CSV file
    Returns:
        List of prompts from the file
    """
    prompts = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check if 'prompt' column exists
            if 'prompt' not in reader.fieldnames:
                print(f"Warning: 'prompt' column not found in {filepath}")
                return prompts
            
            for row in reader:
                prompt = row['prompt']
                prompts.append(prompt)
                
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return prompts


def check_duplicates_across_files(dataset_dir):
    """
    Check for duplicate prompts across the 5 specified CSV files.
    Args:
        dataset_dir: Directory containing the CSV files
    """
    # Define the CSV files to check
    csv_files = [
        "harmbench_prompts.csv",
        "advbench_prompts.csv",
        "sorrybench_prompts.csv",
        "strongreject_prompts.csv",
        "orbench_prompts.csv"
    ]
    
    # Dictionary to store prompts and their source files
    prompt_sources = defaultdict(list)
    
    # Statistics
    total_prompts = 0
    file_prompt_counts = {}
    
    print(f"Checking for duplicates in directory: {dataset_dir}\n")
    print("=" * 60)
    
    # Read prompts from each file
    for csv_file in csv_files:
        filepath = os.path.join(dataset_dir, csv_file)
        print(f"Reading: {csv_file}")
        
        prompts = read_prompts_from_csv(filepath)
        file_prompt_counts[csv_file] = len(prompts)
        total_prompts += len(prompts)
        
        # Track which file each prompt came from
        for prompt in prompts:
            prompt_sources[prompt].append(csv_file)
        
        print(f"  Found {len(prompts)} prompts")
    
    print("=" * 60)
    print(f"\nTotal prompts across all files: {total_prompts}")
    print(f"Unique prompts: {len(prompt_sources)}")
    
    # Find duplicates (prompts that appear in multiple files)
    duplicates = {prompt: sources for prompt, sources in prompt_sources.items() 
                  if len(sources) > 1}
    
    # Find within-file duplicates
    within_file_duplicates = {}
    for csv_file in csv_files:
        filepath = os.path.join(dataset_dir, csv_file)
        prompts = read_prompts_from_csv(filepath)
        
        # Count occurrences of each prompt in this file
        prompt_counts = defaultdict(int)
        for prompt in prompts:
            prompt_counts[prompt] += 1
        
        # Find prompts that appear more than once in this file
        file_duplicates = {prompt: count for prompt, count in prompt_counts.items() 
                          if count > 1}
        
        if file_duplicates:
            within_file_duplicates[csv_file] = file_duplicates
    
    # Report results
    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS RESULTS")
    print("=" * 60)
    
    # Report within-file duplicates
    if within_file_duplicates:
        print("\nWITHIN-FILE DUPLICATES FOUND:")
        for csv_file, duplicates_in_file in within_file_duplicates.items():
            print(f"\n  {csv_file}:")
            for prompt, count in duplicates_in_file.items():
                # Truncate long prompts for display
                display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
                print(f"    - Appears {count} times: \"{display_prompt}\"")
    else:
        print("\n✓ No within-file duplicates found")
    
    # Report cross-file duplicates
    if duplicates:
        print("\nCROSS-FILE DUPLICATES FOUND:")
        print(f"  {len(duplicates)} prompts appear in multiple files\n")
        
        for i, (prompt, sources) in enumerate(duplicates.items(), 1):
            print(f"  Duplicate #{i}:")
            print(f"    Files: {', '.join(sources)}")
            
            # Truncate long prompts for display
            display_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
            print(f"    Prompt: \"{display_prompt}\"")
            print()
    else:
        print("\n✓ No cross-file duplicates found")
    
    # Summary statistics
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files checked: {len(csv_files)}")
    print(f"Total prompts: {total_prompts}")
    print(f"Unique prompts: {len(prompt_sources)}")
    print(f"Cross-file duplicates: {len(duplicates)}")
    print(f"Files with within-file duplicates: {len(within_file_duplicates)}")
    
    # Return status code (0 if no duplicates, 1 if duplicates found)
    return len(duplicates) > 0 or len(within_file_duplicates) > 0


def main():
    parser = argparse.ArgumentParser(
        description="Check for duplicate prompts across multiple CSV files"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/base",
        help="Directory containing the CSV files (default: dataset/base)"
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Directory '{args.dataset_dir}' does not exist")
        return 1
    
    # Run duplicate check
    has_duplicates = check_duplicates_across_files(args.dataset_dir)
    
    # Exit with appropriate code
    if has_duplicates:
        print("\n Duplicates were found. Please review the results above.")
        return 1
    else:
        print("\n No duplicates found across all datasets!")
        return 0


if __name__ == "__main__":
    exit(main())
"""
Master script to filter datasets using both CoT and baseline filtering methods.

This script calls both filter_cot_datasets.py and filter_baseline_datasets.py
using the same pre-computed scores file.

Usage:
    uv -m utils.filter_all_datasets --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                                   --scores_file scores_all_harmful_prompts_cot5_out5.json
"""

import argparse
import os
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter datasets using both CoT and baseline methods"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (for directory structure)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Results parent directory"
    )
    parser.add_argument(
        "--scored_csv",
        type=str,
        required=True,
        help="CSV file containing pre-computed scores"
    )
    
    # CoT filtering parameters
    parser.add_argument(
        "--cot_lower_threshold",
        type=float,
        default=0.05,
        help="CoT lower threshold for creating refusal dataset"
    )
    parser.add_argument(
        "--cot_upper_threshold",
        type=float,
        default=0.6,
        help="CoT upper threshold for creating non-refusal dataset"
    )
    
    # Baseline filtering parameters
    parser.add_argument(
        "--baseline_lower_threshold",
        type=float,
        default=0.05,
        help="Baseline lower threshold for creating refusal dataset"
    )
    parser.add_argument(
        "--baseline_upper_threshold",
        type=float,
        default=0.60,
        help="Baseline upper threshold for creating non-refusal dataset"
    )
    parser.add_argument(
        "--percentage_threshold",
        type=float,
        default=0.75,
        help="Baseline percentage of outputs that must meet threshold"
    )
    
    # Control which filters to run
    parser.add_argument(
        "--run_cot",
        action="store_true",
        default=True,
        help="Run CoT filtering (default: True)"
    )
    parser.add_argument(
        "--run_baseline",
        action="store_true", 
        default=True,
        help="Run baseline filtering (default: True)"
    )
    parser.add_argument(
        "--skip_cot",
        action="store_true",
        help="Skip CoT filtering"
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline filtering"
    )

    return parser.parse_args()


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a subprocess command with error handling.
    
    Args:
        cmd: Command to run as list of strings
        description: Description for logging
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"SUCCESS: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"FAILED: {description} failed with error: {e}")
        return False


def main() -> None:
    # Parse arguments
    args: argparse.Namespace = parse_args()
    
    # Check if scores file exists
    scores_path = os.path.join(args.results_dir, args.model_name, "dataset", args.scored_csv)
    if not os.path.exists(scores_path):
        print(f"Error: Scores file not found at {scores_path}")
        print("Run the scoring script first to generate scores:")
        print(f"python -m utils.score_outputs --model_name {args.model_name}")
        sys.exit(1)
    
    print(f"Using scores file: {scores_path}")
    
    # Determine which filters to run
    run_cot = args.run_cot and not args.skip_cot
    run_baseline = args.run_baseline and not args.skip_baseline
    
    if not run_cot and not run_baseline:
        print("Error: At least one filtering method must be enabled")
        sys.exit(1)
    
    success_count = 0
    total_count = 0
    
    # Run CoT filtering
    if run_cot:
        total_count += 1
        cot_cmd = [
            sys.executable, "-m", "utils.filter_cot_datasets",
            "--model_name", args.model_name,
            "--results_dir", args.results_dir,
            "--scored_csv", args.scored_csv,
            "--lower_threshold", str(args.cot_lower_threshold),
            "--upper_threshold", str(args.cot_upper_threshold)
        ]
        
        if run_command(cot_cmd, "CoT filtering"):
            success_count += 1
    
    # Run baseline filtering
    if run_baseline:
        total_count += 1
        baseline_cmd = [
            sys.executable, "-m", "utils.filter_baseline_datasets",
            "--model_name", args.model_name,
            "--results_dir", args.results_dir,
            "--scored_csv", args.scored_csv,
            "--lower_threshold", str(args.baseline_lower_threshold),
            "--upper_threshold", str(args.baseline_upper_threshold),
            "--percentage_threshold", str(args.percentage_threshold)
        ]
        
        if run_command(baseline_cmd, "Baseline filtering"):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Total filtering methods: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count == total_count:
        print("All filtering operations completed successfully!")
        
        # Show output files
        dataset_dir = os.path.join(args.results_dir, args.model_name, "dataset")
        print(f"\nGenerated files in {dataset_dir}:")
        
        if run_cot:
            print(f"  - refusal_{args.cot_lower_threshold}.csv (CoT refusal)")
            print(f"  - nonrefusal_{args.cot_upper_threshold}.csv (CoT non-refusal)")
        
        if run_baseline:
            print(f"  - refusal_{args.baseline_lower_threshold}_pct{args.percentage_threshold}.csv (Baseline refusal)")
            print(f"  - nonrefusal_{args.baseline_upper_threshold}_pct{args.percentage_threshold}.csv (Baseline non-refusal)")
            
    else:
        print("Some filtering operations failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
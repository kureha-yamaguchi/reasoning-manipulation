"""
Print mean and standard deviation of the output generation score (strongreject_score)
across rollouts for each CoT number, for a given prompt index.

Covers two file sets:
  - Base model:     scored_resampling_results_idx{prompt_index}_cot{n}.csv
  - Transfer model: scored_transfer_results_idx{prompt_index}_cot{n}_transfer_from_{base_model_short}.csv

Typical usage
-------------
uv run -m utils.heuristic.print_rollout_stats \\
    --prompt_index 806 \\
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --transfer_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print mean/std of strongreject_score per CoT for a given prompt index"
    )
    parser.add_argument("--prompt_index", type=int, required=True,
                        help="Original prompt index")
    parser.add_argument("--base_model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model whose CoT first sentences are resampled")
    parser.add_argument("--transfer_model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Model on which the prefill attack is applied (optional)")
    parser.add_argument("--results_dir", type=str, default="results/",
                        help="Root results directory")
    return parser.parse_args()


def print_stats(label: str, paths: list[str]) -> None:
    if not paths:
        print(f"  [no files found]\n")
        return

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.dropna(subset=["strongreject_score"])
        scores = df["strongreject_score"].values
        # Extract cot number from filename
        basename = os.path.basename(path)
        # filename patterns:
        #   scored_resampling_results_idx{idx}_cot{n}.csv
        #   scored_transfer_results_idx{idx}_cot{n}_transfer_from_*.csv
        try:
            cot_part = [p for p in basename.split("_") if p.startswith("cot")][0]
            cot_n = int(cot_part[3:].split(".")[0])
        except (IndexError, ValueError):
            cot_n = -1
        rows.append((cot_n, scores))

    rows.sort(key=lambda x: x[0])

    print(f"{label}")
    print(f"  {'CoT':>5}  {'N':>4}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*5}  {'-'*4}  {'-'*8}  {'-'*8}")
    for cot_n, scores in rows:
        n = len(scores)
        mean = float(np.mean(scores)) if n > 0 else float("nan")
        std = float(np.std(scores, ddof=1)) if n > 1 else float("nan")
        print(f"  {cot_n:>5}  {n:>4}  {mean:>8.4f}  {std:>8.4f}")
    print()


def main():
    args = parse_args()

    base_model_short = args.base_model.split("/")[-1]

    # Base model files
    base_pattern = os.path.join(
        args.results_dir,
        args.base_model,
        "dataset",
        "resample",
        f"scored_resampling_results_idx{args.prompt_index}_cot*.csv",
    )
    base_paths = sorted(glob.glob(base_pattern))

    print(f"Prompt index: {args.prompt_index}\n")

    print(f"Base model: {args.base_model}")
    print(f"Pattern: {base_pattern}")
    print(f"Found {len(base_paths)} file(s)")
    print_stats("  CoT stats (base model):", base_paths)

    # Transfer model files
    transfer_pattern = os.path.join(
        args.results_dir,
        args.transfer_model,
        "dataset",
        "resample",
        f"scored_transfer_results_idx{args.prompt_index}_cot*_transfer_from_{base_model_short}.csv",
    )
    transfer_paths = sorted(glob.glob(transfer_pattern))

    print(f"Transfer model: {args.transfer_model}")
    print(f"Pattern: {transfer_pattern}")
    print(f"Found {len(transfer_paths)} file(s)")
    print_stats("  CoT stats (transfer model):", transfer_paths)


if __name__ == "__main__":
    main()

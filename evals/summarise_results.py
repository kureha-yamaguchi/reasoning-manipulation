#!/usr/bin/env python3
"""Summarise Inspect eval results into a comparison table.

Reads Inspect JSON log files from results/{model_label}/eval_results/
and produces a base-vs-ortho comparison table for each model family
and benchmark.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model configs: (label, results_subdir) grouped by family
MODEL_FAMILIES = {
    "Qwen3-8B": [
        ("base", "Qwen/Qwen3-8B"),
        ("ortho", "Qwen/Qwen3-8B/ortho_model_cot_layer_17"),
    ],
    "gpt-oss-20b": [
        ("base", "openai/gpt-oss-20b"),
        ("ortho", "openai/gpt-oss-20b/ortho_model_cot_layer_19"),
    ],
}

BENCHMARKS = ["aime2025", "gpqa_diamond", "math"]

# Map benchmark names to task identifiers that appear in Inspect logs
TASK_IDENTIFIERS = {
    "aime2025": "aime2025",
    "gpqa_diamond": "gpqa_diamond",
    "math": "mathematics",
}


def find_log_files(eval_results_dir: Path) -> list[Path]:
    """Find all Inspect JSON log files in a directory."""
    logs = []
    if eval_results_dir.exists():
        logs.extend(eval_results_dir.glob("*.json"))
        logs.extend(eval_results_dir.glob("**/*.eval"))
        logs.extend(eval_results_dir.glob("**/*.json"))
    return sorted(set(logs))


def extract_score_from_log(log_path: Path) -> dict | None:
    """Extract task name and score from an Inspect log file."""
    try:
        with open(log_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Inspect logs have a 'results' key with 'scores'
    results = data.get("results", {})
    eval_info = data.get("eval", {})
    task = eval_info.get("task", "")

    scores = results.get("scores", [])
    if not scores:
        return None

    # Get the primary metric from the first scorer
    primary_score = scores[0]
    metrics = primary_score.get("metrics", {})

    # Try common metric names
    for metric_name in ["accuracy", "mean", "correct", "score"]:
        if metric_name in metrics:
            value = metrics[metric_name].get("value")
            if value is not None:
                return {
                    "task": task,
                    "metric": metric_name,
                    "value": value,
                }

    # Fall back to first available metric
    if metrics:
        first_key = next(iter(metrics))
        value = metrics[first_key].get("value")
        if value is not None:
            return {
                "task": task,
                "metric": first_key,
                "value": value,
            }

    return None


def match_benchmark(task_name: str, benchmark: str) -> bool:
    """Check if a task name matches a benchmark."""
    identifier = TASK_IDENTIFIERS.get(benchmark, benchmark)
    return identifier.lower() in task_name.lower()


def collect_results() -> dict:
    """Collect all results into a structured dict.

    Returns: {family: {variant: {benchmark: score}}}
    """
    all_results = {}

    for family, variants in MODEL_FAMILIES.items():
        all_results[family] = {}
        for variant_name, label in variants:
            eval_dir = PROJECT_ROOT / "results" / label / "eval_results"
            variant_scores = {}

            logs = find_log_files(eval_dir)
            for log_path in logs:
                info = extract_score_from_log(log_path)
                if info is None:
                    continue
                for benchmark in BENCHMARKS:
                    if match_benchmark(info["task"], benchmark):
                        variant_scores[benchmark] = info["value"]
                        break

            all_results[family][variant_name] = variant_scores

    return all_results


def print_table(results: dict):
    """Print a formatted comparison table."""
    header = f"{'Model':<20} {'Variant':<10}"
    for bench in BENCHMARKS:
        header += f" {bench:>15}"
    print(header)
    print("-" * len(header))

    for family in MODEL_FAMILIES:
        for variant in ["base", "ortho"]:
            scores = results.get(family, {}).get(variant, {})
            row = f"{family:<20} {variant:<10}"
            for bench in BENCHMARKS:
                val = scores.get(bench)
                if val is not None:
                    row += f" {val:>15.4f}"
                else:
                    row += f" {'N/A':>15}"
            print(row)

        # Print delta row
        base_scores = results.get(family, {}).get("base", {})
        ortho_scores = results.get(family, {}).get("ortho", {})
        delta_row = f"{'':<20} {'delta':<10}"
        for bench in BENCHMARKS:
            b = base_scores.get(bench)
            o = ortho_scores.get(bench)
            if b is not None and o is not None:
                delta = o - b
                delta_row += f" {delta:>+15.4f}"
            else:
                delta_row += f" {'N/A':>15}"
        print(delta_row)
        print()


def save_csv(results: dict, output_path: Path):
    """Save results as CSV."""
    with open(output_path, "w") as f:
        f.write("model_family,variant," + ",".join(BENCHMARKS) + "\n")
        for family in MODEL_FAMILIES:
            for variant in ["base", "ortho"]:
                scores = results.get(family, {}).get(variant, {})
                values = []
                for bench in BENCHMARKS:
                    val = scores.get(bench)
                    values.append(f"{val:.4f}" if val is not None else "")
                f.write(f"{family},{variant},{','.join(values)}\n")
    print(f"\nCSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarise Inspect eval results")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save results as CSV",
    )
    args = parser.parse_args()

    results = collect_results()

    print("\n" + "=" * 70)
    print("Off-Task Eval Results: Base vs Orthogonalized")
    print("=" * 70 + "\n")

    print_table(results)

    if args.csv:
        save_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()

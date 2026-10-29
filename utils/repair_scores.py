"""
Script to repair NaN scores in already-scored CSV files by re-calling the StrongReject evaluator.

Reads the scored CSV, identifies rows with NaN scores, re-evaluates them (up to --max_retries
attempts per row), and writes the repaired scores back into the same file.

====================
Example usage
====================

    STRONGREJECT_VLLM_URL=http://localhost:8000/v1 \
    uv run -m utils.repair_scores \
    --scored_csv results/Qwen/Qwen3-8B/dataset/scored_train_harmful_prompts_cot5_out5.csv

"""

import csv
import math
import os
import argparse
from typing import List, Dict

from strong_reject.evaluate import evaluate


SCORE_COLUMNS = {
    "score": "strongreject_score",
    "refusal": "strongreject_refusal",
    "convincingness": "strongreject_convincingness",
    "specificity": "strongreject_specificity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair NaN scores in scored CSV files by re-calling the evaluator"
    )
    parser.add_argument(
        "--scored_csv",
        type=str,
        required=True,
        help="Path to the scored CSV file to repair"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Maximum number of retry attempts per row (default: 10)"
    )
    parser.add_argument(
        "--extra_csvs",
        type=str,
        nargs="*",
        default=[],
        help="Further scored CSVs to repair in the same run (with --threads)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="If >0, repair NaN rows of all given files concurrently with this many threads"
    )
    return parser.parse_args()


def is_nan_score(value: str) -> bool:
    """Check if a score value is NaN or missing."""
    if value is None or value.strip() == "":
        return True
    try:
        return math.isnan(float(value))
    except (ValueError, TypeError):
        return True


def load_scored_csv(csv_path: str) -> tuple[List[Dict[str, str]], List[str]]:
    """Load the scored CSV and return rows + fieldnames."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return rows, fieldnames


def find_nan_rows(rows: List[Dict[str, str]]) -> List[int]:
    """Find indices of rows where strongreject_score is NaN."""
    nan_indices = []
    for i, row in enumerate(rows):
        score_val = row.get("strongreject_score", "")
        if is_nan_score(score_val):
            nan_indices.append(i)
    return nan_indices


def repair_row(row: Dict[str, str], max_retries: int) -> Dict[str, float] | None:
    """
    Re-evaluate a single row using strongreject_rubric.

    Returns dict of {score_key: value} on success, or None if all retries fail.
    """
    prompt = row.get("prompt", "")
    output = row.get("output", "")

    for attempt in range(1, max_retries + 1):
        try:
            results = evaluate(prompt, output, ["strongreject_rubric"])
            result = results[0]  # single evaluator

            # Check if the score is valid (not NaN)
            score = result.get("score")
            if score is not None and not math.isnan(float(score)):
                return result
        except Exception as e:
            print(f"    Attempt {attempt} error: {e}")

    return None


def save_csv(rows: List[Dict[str, str]], fieldnames: List[str], csv_path: str) -> None:
    """Write rows back to the CSV file."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repair_threaded(csv_paths: List[str], max_retries: int, threads: int) -> None:
    """Same per-row retries as the serial path, run concurrently across all files; atomic writes."""
    from concurrent.futures import ThreadPoolExecutor
    from strong_reject.vllm_server import ensure_vllm_server

    ensure_vllm_server()  # initialise the shared server handle before threads use it
    files = {p: load_scored_csv(p) for p in csv_paths}
    tasks = [(p, i) for p, (rows, _) in files.items() for i in find_nan_rows(rows)]
    print(f"Repairing {len(tasks)} NaN rows across {len(files)} files with {threads} threads")
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(lambda t: repair_row(files[t[0]][0][t[1]], max_retries), tasks))
    for (p, i), result in zip(tasks, results):
        rows, fieldnames = files[p]
        if result is not None:
            for key, csv_col in SCORE_COLUMNS.items():
                if key in result and csv_col in fieldnames:
                    rows[i][csv_col] = result[key]
    for p, (rows, fieldnames) in files.items():
        save_csv(rows, fieldnames, p + ".tmp")
        os.replace(p + ".tmp", p)
    repaired = sum(r is not None for r in results)
    print(f"Repaired: {repaired}/{len(tasks)}")
    print(f"Failed:   {len(tasks) - repaired}/{len(tasks)}")


def main() -> None:
    args = parse_args()

    if args.threads > 0:
        repair_threaded([args.scored_csv] + args.extra_csvs, args.max_retries, args.threads)
        return

    if not os.path.exists(args.scored_csv):
        print(f"Error: File not found: {args.scored_csv}")
        return

    # Load data
    rows, fieldnames = load_scored_csv(args.scored_csv)
    nan_indices = find_nan_rows(rows)

    print(f"Loaded {len(rows)} rows from {args.scored_csv}")
    print(f"Found {len(nan_indices)} rows with NaN scores")

    if not nan_indices:
        print("Nothing to repair!")
        return

    # Repair each NaN row
    repaired = 0
    failed = 0

    for count, idx in enumerate(nan_indices, 1):
        row = rows[idx]
        prompt_preview = row.get("prompt", "")[:80]
        print(f"[{count}/{len(nan_indices)}] Repairing row {idx}: {prompt_preview}...")

        result = repair_row(row, args.max_retries)

        if result is not None:
            # Write the repaired scores back into the row
            for key, csv_col in SCORE_COLUMNS.items():
                if key in result and csv_col in fieldnames:
                    rows[idx][csv_col] = result[key]
            repaired += 1
            print(f"    -> score={result.get('score', 'N/A')}")
        else:
            failed += 1
            print(f"    -> FAILED after {args.max_retries} retries")

        # Periodically save progress (every 50 repairs)
        if count % 50 == 0:
            print(f"  Saving progress ({count}/{len(nan_indices)})...")
            save_csv(rows, fieldnames, args.scored_csv)

    # Final save
    save_csv(rows, fieldnames, args.scored_csv)

    print(f"\nRepair complete!")
    print(f"  Repaired: {repaired}/{len(nan_indices)}")
    print(f"  Failed:   {failed}/{len(nan_indices)}")
    print(f"  Saved to: {args.scored_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot base vs ortho eval results as grouped bar charts with error bars."""

import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model configs: family -> [(variant_label, results_subdir)]
MODEL_FAMILIES = {
    "Qwen3-8B": [
        ("Base", "Qwen/Qwen3-8B"),
        ("Ortho (layer 17)", "Qwen/Qwen3-8B/ortho_model_cot_layer_17"),
    ],
    "GPT-oss-20B": [
        ("Base", "openai/gpt-oss-20b"),
        ("Ortho (layer 19)", "openai/gpt-oss-20b/ortho_model_cot_layer_19"),
    ],
}

BENCHMARKS = ["aime2025", "gpqa_diamond"]
BENCHMARK_LABELS = {"aime2025": "AIME 2025", "gpqa_diamond": "GPQA Diamond"}
TASK_IDENTIFIERS = {"aime2025": "aime2025", "gpqa_diamond": "gpqa_diamond"}


def extract_scores(eval_dir: Path) -> dict[str, dict]:
    """Extract accuracy & stderr from the latest successful .eval file per benchmark.

    Returns: {benchmark: {"accuracy": float, "stderr": float}}
    """
    if not eval_dir.exists():
        return {}

    # Sort by filename (timestamp) descending so we check newest first
    eval_files = sorted(eval_dir.glob("*.eval"), reverse=True)
    found: dict[str, dict] = {}

    for ef in eval_files:
        try:
            with zipfile.ZipFile(ef) as z:
                with z.open("header.json") as jf:
                    data = json.load(jf)
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            continue

        if data.get("status") != "success":
            continue

        task = data.get("eval", {}).get("task", "")
        for bench, ident in TASK_IDENTIFIERS.items():
            if bench in found:
                continue
            if ident.lower() in task.lower():
                scores = data.get("results", {}).get("scores", [])
                if not scores:
                    continue
                metrics = scores[0].get("metrics", {})
                acc = metrics.get("accuracy", {}).get("value")
                se = metrics.get("stderr", {}).get("value")
                if acc is not None:
                    found[bench] = {"accuracy": acc, "stderr": se or 0.0}
                break

        if len(found) == len(BENCHMARKS):
            break

    return found


def main():
    # Collect all data
    data = {}  # {family: {variant_label: {benchmark: {accuracy, stderr}}}}
    for family, variants in MODEL_FAMILIES.items():
        data[family] = {}
        for variant_label, subdir in variants:
            eval_dir = PROJECT_ROOT / "results" / subdir / "eval_results"
            data[family][variant_label] = extract_scores(eval_dir)

    # Print table
    print(f"\n{'Model':<20} {'Variant':<20} {'Benchmark':<18} {'Accuracy':>10} {'Stderr':>10}")
    print("-" * 80)
    for family in MODEL_FAMILIES:
        for variant_label, _ in MODEL_FAMILIES[family]:
            scores = data[family].get(variant_label, {})
            for bench in BENCHMARKS:
                s = scores.get(bench, {})
                acc = s.get("accuracy")
                se = s.get("stderr")
                acc_str = f"{acc:.4f}" if acc is not None else "N/A"
                se_str = f"{se:.4f}" if se is not None else "N/A"
                print(f"{family:<20} {variant_label:<20} {BENCHMARK_LABELS[bench]:<18} {acc_str:>10} {se_str:>10}")
        print()

    # --- Plot: one subplot per model family ---
    fig, axes = plt.subplots(1, len(MODEL_FAMILIES), figsize=(5 * len(MODEL_FAMILIES), 5),
                              sharey=False)
    if len(MODEL_FAMILIES) == 1:
        axes = [axes]

    colors = {"Base": "#4C72B0", "default": "#DD8452"}
    bar_width = 0.3

    for ax, (family, variants) in zip(axes, MODEL_FAMILIES.items()):
        variant_labels = [v[0] for v in variants]
        x = np.arange(len(BENCHMARKS))

        for i, vlabel in enumerate(variant_labels):
            scores = data[family].get(vlabel, {})
            accs = [scores.get(b, {}).get("accuracy", 0) for b in BENCHMARKS]
            errs = [scores.get(b, {}).get("stderr", 0) for b in BENCHMARKS]
            color = colors.get(vlabel, colors["default"])

            bars = ax.bar(x + i * bar_width, accs, bar_width, yerr=errs,
                          label=vlabel, color=color, capsize=4, edgecolor="white",
                          linewidth=0.5)

            # Add value labels on bars
            for bar, acc, err in zip(bars, accs, errs):
                if acc > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                            f"{acc:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(family, fontsize=13, fontweight="bold")
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels([BENCHMARK_LABELS[b] for b in BENCHMARKS], fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Off-Task Eval: Base vs Orthogonalized", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "results" / "eval_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")

    # Also save PDF for paper
    pdf_path = PROJECT_ROOT / "results" / "eval_comparison.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to:  {pdf_path}")


if __name__ == "__main__":
    main()

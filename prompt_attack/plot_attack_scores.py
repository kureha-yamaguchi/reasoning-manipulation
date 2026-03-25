"""Plot GCG+IRIS attack scores across models and beta values.

Generates:
  1. Bar chart: mean StrongREJECT score per model & beta, with counterfactual baseline
  2. Violin plot: score distributions per model & beta
  3. Line plot: mean score vs beta for each model

Usage:
    uv run python -m prompt_attack.plot_attack_scores
    uv run python -m prompt_attack.plot_attack_scores --refusal-mode baseline
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from prompt_attack.models import MODEL_REGISTRY, get_model_config
from prompt_attack.score_rollouts import (
    BETAS,
    get_counterfactual_csv_path,
    get_scored_rollout_csv_path,
)

DISPLAY_NAMES = {
    "deepseek-llama-8b": "DS-R1-Llama-8B",
    "deepseek-qwen-7b": "DS-R1-Qwen-7B",
    "qwen3-8b": "Qwen3-8B",
    "gpt-oss-20b": "GPT-OSS-20B",
}

BETA_PALETTE = {
    "Counterfactual": "#999999",
    "β=0.0": "#2166ac",
    "β=0.3": "#67a9cf",
    "β=0.5": "#fddbc7",
    "β=0.7": "#ef8a62",
    "β=1.0": "#b2182b",
}

MODEL_PALETTE = {
    "DS-R1-Llama-8B": "#1f78b4",
    "DS-R1-Qwen-7B": "#33a02c",
    "Qwen3-8B": "#e31a1c",
    "GPT-OSS-20B": "#ff7f00",
}


def load_scores(model_alias: str, mode: str, run_tag: str = "") -> pd.DataFrame:
    """Load scored rollouts for all betas + counterfactual for one model."""
    rows = []

    # Counterfactual (unattacked baseline)
    cf_path = get_counterfactual_csv_path(model_alias, mode, run_tag)
    if cf_path.exists():
        cf = pd.read_csv(cf_path)
        cf = cf[["prompt", "strongreject_score"]].dropna()
        cf["beta_label"] = "Counterfactual"
        cf["beta"] = -1.0
        cf["model"] = DISPLAY_NAMES[model_alias]
        rows.append(cf[["prompt", "strongreject_score", "beta_label", "beta", "model"]])

    # Attacked rollouts per beta
    for beta in BETAS:
        scored_path = get_scored_rollout_csv_path(model_alias, beta, mode, run_tag)
        if not scored_path.exists():
            print(f"  WARN: missing {scored_path.name}")
            continue
        df = pd.read_csv(scored_path, usecols=["prompt", "strongreject_score"]).dropna()
        df["beta_label"] = f"β={beta:.1f}"
        df["beta"] = beta
        df["model"] = DISPLAY_NAMES[model_alias]
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_bar_chart(data: pd.DataFrame, output_dir: str, mode: str, tag: str = ""):
    """Bar chart: mean StrongREJECT score per model and beta."""
    hue_order = ["Counterfactual"] + [f"β={b:.1f}" for b in BETAS]
    hue_order = [h for h in hue_order if h in data["beta_label"].unique()]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=data, x="model", y="strongreject_score",
        hue="beta_label", hue_order=hue_order, palette=BETA_PALETTE,
        errorbar="ci", capsize=0.05, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean StrongREJECT Score")
    ax.set_title(f"GCG+IRIS Attack Scores by Model and β ({mode} refusal direction)")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax_right = ax.secondary_yaxis("right")
    ax_right.set_yticks([])
    ax.annotate("(comply)", xy=(1.01, 1), xycoords="axes fraction",
                fontsize=10, fontstyle="italic", va="top")
    ax.annotate("(refuse)", xy=(1.01, 0), xycoords="axes fraction",
                fontsize=10, fontstyle="italic", va="bottom")

    plt.tight_layout()
    path = f"{output_dir}/attack_barplot_{mode}{tag}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_violin(data: pd.DataFrame, output_dir: str, mode: str, tag: str = ""):
    """Violin plot: score distributions per model and beta."""
    hue_order = ["Counterfactual"] + [f"β={b:.1f}" for b in BETAS]
    hue_order = [h for h in hue_order if h in data["beta_label"].unique()]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(
        data=data, x="model", y="strongreject_score",
        hue="beta_label", hue_order=hue_order, palette=BETA_PALETTE,
        cut=0, inner="quartile", density_norm="width", gap=0.15, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("StrongREJECT Score")
    ax.set_title(f"GCG+IRIS Score Distributions by Model and β ({mode} refusal direction)")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = f"{output_dir}/attack_violin_{mode}{tag}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_line(data: pd.DataFrame, output_dir: str, mode: str, tag: str = ""):
    """Line plot: mean score vs beta for each model (attack scores only)."""
    attacked = data[data["beta"] >= 0].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in attacked.groupby("model"):
        means = group.groupby("beta")["strongreject_score"].mean()
        sems = group.groupby("beta")["strongreject_score"].sem()
        color = MODEL_PALETTE.get(model_name, None)
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    marker="o", capsize=4, label=model_name, color=color)

    # Counterfactual baselines as horizontal dashed lines
    cf = data[data["beta"] < 0]
    for model_name, group in cf.groupby("model"):
        mean_cf = group["strongreject_score"].mean()
        color = MODEL_PALETTE.get(model_name, None)
        ax.axhline(mean_cf, color=color, linestyle="--", alpha=0.5)

    ax.set_xlabel("β  (0 = pure GCG, 1 = pure IRIS)")
    ax.set_ylabel("Mean StrongREJECT Score")
    ax.set_title(f"Attack Success vs β ({mode} refusal direction)")
    ax.set_xticks(BETAS)
    ax.legend(title="Model")
    plt.tight_layout()
    path = f"{output_dir}/attack_line_{mode}{tag}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GCG+IRIS attack scores")
    parser.add_argument("--refusal-mode", type=str, default="cot",
                        choices=["cot", "baseline"])
    parser.add_argument("--output-dir", type=str, default="figures/attack")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Tag appended to filenames to separate runs (e.g. 'v2')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_data = []
    for alias in MODEL_REGISTRY:
        print(f"Loading {alias}...")
        df = load_scores(alias, args.refusal_mode, args.run_tag)
        if not df.empty:
            all_data.append(df)
        else:
            print(f"  No data for {alias}")

    if not all_data:
        print("No data found.")
        return

    data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows: {len(data):,}")

    # Print summary table
    summary = (
        data.groupby(["model", "beta_label"])["strongreject_score"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(f"\n{summary}\n")

    tag = f"_{args.run_tag}" if args.run_tag else ""
    plot_bar_chart(data, args.output_dir, args.refusal_mode, tag)
    plot_violin(data, args.output_dir, args.refusal_mode, tag)
    plot_line(data, args.output_dir, args.refusal_mode, tag)

    print("Done.")


if __name__ == "__main__":
    main()

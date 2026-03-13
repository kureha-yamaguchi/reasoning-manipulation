"""
Generate box plots, violin plots, and layered histograms comparing StrongReject
scores before and after intervention.

This module creates three separate plots to visualize the distribution of
strongreject_score values from baseline test data versus intervention results.

Example usage:
    uv run -m utils.plot_boxplot_comparison \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --type cot \
        --layer 17
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate box plot comparing strongreject_score before and after intervention"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
    parser.add_argument("--type", type=str, required=True,
                        help="Type of intervention (e.g., cot, baseline, all)")
    parser.add_argument("--layer", type=str, required=True,
                        help="Layer number (e.g., 16)")
    parser.add_argument("--harmless", action='store_true',
                        help='For harmless nonrefusal, harmful refusal dataset configuration')
    return parser.parse_args()


def make_plots(comparison_data, order, palette, title_suffix, output_dir, file_suffix):
    """Generate box plot, violin plot, and layered histogram for the given data."""
    n_groups = len(order)
    ylim = (-0.5, n_groups - 0.5)

    annotations = [
        ('(comply)', (1, 1.01), 'right'),
        ('(refuse)', (0, 1.01), 'left'),
    ]

    # --- Box plot ---
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Score', y='Group', data=comparison_data, width=0.5,
                     order=order, palette=palette, showfliers=False)
    ax.set_ylim(*ylim)
    plt.title(f'StrongReject Score (Box) — {title_suffix}', fontsize=14)
    plt.xlabel('StrongReject Score', fontsize=12)
    plt.ylabel('')
    for text, xy, ha in annotations:
        ax.annotate(text, xy=xy, xycoords='axes fraction', fontsize=10, fontstyle='italic', ha=ha)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_comparison_{file_suffix}.png", dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {output_dir}/boxplot_comparison_{file_suffix}.png")
    plt.close()

    # --- Violin plot ---
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Score', y='Group', data=comparison_data, width=0.7,
                        order=order, palette=palette, inner='quartile', cut=0)
    ax.set_ylim(*ylim)
    plt.title(f'StrongReject Score (Violin) — {title_suffix}', fontsize=14)
    plt.xlabel('StrongReject Score', fontsize=12)
    plt.ylabel('')
    for text, xy, ha in annotations:
        ax.annotate(text, xy=xy, xycoords='axes fraction', fontsize=10, fontstyle='italic', ha=ha)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/violin_comparison_{file_suffix}.png", dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to: {output_dir}/violin_comparison_{file_suffix}.png")
    plt.close()

    # --- KDE plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for group in order:
        scores = comparison_data[comparison_data['Group'] == group]['Score']
        sns.kdeplot(scores, label=group, color=palette[group], fill=True, alpha=0.3, linewidth=2, clip=(0, 1), ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title(f'StrongReject Score (Density) — {title_suffix}', fontsize=14)
    ax.set_xlabel('StrongReject Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.annotate('(refuse)', xy=(0, -0.06), xycoords='axes fraction', fontsize=10, fontstyle='italic', ha='left')
    ax.annotate('(comply)', xy=(1, -0.06), xycoords='axes fraction', fontsize=10, fontstyle='italic', ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kde_comparison_{file_suffix}.png", dpi=300, bbox_inches='tight')
    print(f"KDE plot saved to: {output_dir}/kde_comparison_{file_suffix}.png")
    plt.close()


def main():
    args = parse_args()

    before_path = f"results/{args.model_name}/dataset/scored_test_harmful_prompts_cot5_out5.csv"

    output_dir = f"results/{args.model_name}/figures"
    os.makedirs(output_dir, exist_ok=True)

    if args.type == 'all':
        layers = args.layer.split(',')
        cot_layer = layers[0].strip()
        baseline_layer = layers[1].strip()

        if args.harmless:
            cot_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}_harmless.csv"
            baseline_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layer}_harmless.csv"
            file_suffix = f"{args.type}_layer_{cot_layer}_{baseline_layer}_harmless"
        else:
            cot_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}.csv"
            baseline_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layer}.csv"
            file_suffix = f"{args.type}_layer_{cot_layer}_{baseline_layer}"

        before_df = pd.read_csv(before_path)
        cot_df = pd.read_csv(cot_path)
        baseline_df = pd.read_csv(baseline_path)

        before_scores = before_df['strongreject_score'].dropna()
        cot_scores = cot_df['strongreject_score'].dropna()
        baseline_scores = baseline_df['strongreject_score'].dropna()

        comparison_data = pd.DataFrame({
            'Score': list(before_scores) + list(cot_scores) + list(baseline_scores),
            'Group': ['Before'] * len(before_scores) + ['After (cot)'] * len(cot_scores) + ['After (baseline)'] * len(baseline_scores)
        })
        comparison_data['Score'] = comparison_data['Score'].clip(lower=0, upper=1)

        order = ['After (cot)', 'After (baseline)', 'Before']
        palette = {'After (baseline)': '#ff9900', 'After (cot)': '#e85d75', 'Before': '#50a9a9'}
        title_suffix = (
            f"{args.type.upper()}, Layer (baseline: {baseline_layer}, cot: {cot_layer})\n"
            f"Model: {args.model_name}\n"
            f"Rollouts (5 cot 5 output) per prompt in holdout set (487 harmful prompts)"
        )

        make_plots(comparison_data, order, palette, title_suffix, output_dir, file_suffix)

    else:
        after_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_{args.type}_layer_{args.layer}.csv"
        file_suffix = f"{args.type}_layer_{args.layer}"

        before_df = pd.read_csv(before_path)
        after_df = pd.read_csv(after_path)

        before_scores = before_df['strongreject_score'].dropna()
        after_scores = after_df['strongreject_score'].dropna()

        comparison_data = pd.DataFrame({
            'Score': list(before_scores) + list(after_scores),
            'Group': ['Before'] * len(before_scores) + ['After'] * len(after_scores)
        })
        comparison_data['Score'] = comparison_data['Score'].clip(lower=0, upper=1)

        order = ['After', 'Before']
        palette = {'Before': '#50a9a9', 'After': '#ff9900'}
        title_suffix = (
            f"{args.type.upper()}, Layer {args.layer}\n"
            f"Model: {args.model_name}\n"
            f"Rollouts (5 cot 5 output) per prompt in holdout set (487 harmful prompts)"
        )

        make_plots(comparison_data, order, palette, title_suffix, output_dir, file_suffix)


if __name__ == "__main__":
    main()

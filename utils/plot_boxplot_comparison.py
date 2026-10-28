"""
Generate box plots, violin plots, and layered histograms comparing StrongREJECT
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
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate box plot comparing strongreject_score before and after intervention"
    )
    parser.add_argument("--model_name", type=str,
                        help="Model name (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
    parser.add_argument("--type", type=str,
                        help="Type of intervention (e.g., cot, baseline, all)")
    parser.add_argument("--layer", type=str,
                        help="Layer number (e.g., 16)")
    parser.add_argument("--harmless", action='store_true',
                        help='For harmless nonrefusal, harmful refusal dataset configuration')
    parser.add_argument("--poster", action='store_true',
                        help='Generate simplified poster plot (all models, before vs after CoT, harmful dir)')
    parser.add_argument("--poster-baseline", action='store_true',
                        help='Generate simplified poster plot (all models, before vs after end-of-prompt, harmful dir)')
    parser.add_argument("--poster-both", action='store_true',
                        help='Generate poster plot with both CoT and end-of-prompt interventions (hard subset)')
    parser.add_argument("--poster-both-full", action='store_true',
                        help='Generate poster plot with both interventions on full holdout set')
    parser.add_argument("--zero-pad", action='store_true',
                        help='Zero-pad missing generations to account for non-completed CoTs')
    args = parser.parse_args()
    if not args.poster and not args.poster_baseline and not args.poster_both and not args.poster_both_full and not (args.model_name and args.type and args.layer):
        parser.error("--model_name, --type, and --layer are required unless --poster, --poster-baseline, --poster-both, or --poster-both-full is used")
    return args


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
    plt.title(f'StrongREJECT Score (Box) — {title_suffix}', fontsize=14)
    plt.xlabel('StrongREJECT Score', fontsize=12)
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
    plt.title(f'StrongREJECT Score (Violin) — {title_suffix}', fontsize=14)
    plt.xlabel('StrongREJECT Score', fontsize=12)
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
    ax.set_title(f'StrongREJECT Score (Density) — {title_suffix}', fontsize=14)
    ax.set_xlabel('StrongREJECT Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.annotate('(refuse)', xy=(0, -0.06), xycoords='axes fraction', fontsize=10, fontstyle='italic', ha='left')
    ax.annotate('(comply)', xy=(1, -0.06), xycoords='axes fraction', fontsize=10, fontstyle='italic', ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kde_comparison_{file_suffix}.png", dpi=300, bbox_inches='tight')
    print(f"KDE plot saved to: {output_dir}/kde_comparison_{file_suffix}.png")
    plt.close()


MODEL_NAMES = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen3-8B",
    "openai/gpt-oss-20b",
]
DISPLAY_NAMES = [
    "DeepSeek-R1-\nDistill-Llama-8B",
    "DeepSeek-R1-\nDistill-Qwen-7B",
    "Qwen3-8B",
    "GPT-OSS-20B",
]
REFUSAL_THRESHOLD = 0.4
N_REPS = 5 * 5  # cot_reps * output_reps


def pad_scores_with_zeros(scores, expected):
    """Zero-pad scores to account for non-completed generations."""
    n_missing = expected - len(scores)
    if n_missing > 0:
        return pd.concat([scores, pd.Series(np.zeros(n_missing))], ignore_index=True)
    return scores


def get_common_refused_prompts():
    """Find prompts with max score < REFUSAL_THRESHOLD across all models (using before data)."""
    per_model_refused = {}
    before_dfs = {}
    for model_name in MODEL_NAMES:
        before_path = f"results/{model_name}/dataset/scored_test_harmful_prompts_cot5_out5.csv"
        before_dfs[model_name] = pd.read_csv(before_path)
        prompt_max = before_dfs[model_name].groupby('prompt')['strongreject_score'].max()
        per_model_refused[model_name] = set(prompt_max[prompt_max < REFUSAL_THRESHOLD].index)

    common_refused = per_model_refused[MODEL_NAMES[0]]
    for m in MODEL_NAMES[1:]:
        common_refused = common_refused & per_model_refused[m]
    print(f"Common always-refused prompts (threshold={REFUSAL_THRESHOLD}): {len(common_refused)}")
    return common_refused, before_dfs


def style_poster_ax(ax):
    ax.set_xlabel('')
    ax.set_ylabel('StrongREJECT Score', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='', fontsize=11, loc='center right',
              bbox_to_anchor=(1.32, 0.5))
    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks([])
    ax_right.set_ylabel('')
    ax.annotate('(comply)', xy=(1.01, 1), xycoords='axes fraction', fontsize=11, fontstyle='italic', va='top')
    ax.annotate('(refuse)', xy=(1.01, 0), xycoords='axes fraction', fontsize=11, fontstyle='italic', va='bottom')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.3)


def _render_poster_plots(combined_df, condition_order, palette, output_dir, file_tag):
    """Render box plot, violin plot, and KDE grid for poster data."""
    # Box plot
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.boxplot(data=combined_df, x='Model', y='Score', hue='Condition',
                hue_order=condition_order, palette=palette, showfliers=False,
                width=0.6, linewidth=1.5, ax=ax)
    style_poster_ax(ax)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f"{output_dir}/boxplot_{file_tag}.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_dir}/boxplot_{file_tag}.png/pdf")
    plt.close()

    # Violin plot
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.violinplot(data=combined_df, x='Model', y='Score', hue='Condition',
                   hue_order=condition_order, palette=palette,
                   cut=0, inner='quartile', density_norm='width', gap=0.15, ax=ax)
    style_poster_ax(ax)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f"{output_dir}/violin_{file_tag}.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_dir}/violin_{file_tag}.png/pdf")
    plt.close()

    # KDE grid (one subplot per model)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    display_flat = [dn.replace('\n', ' ') for dn in DISPLAY_NAMES]
    for ax, display_name, display_flat_name in zip(axes.flat, DISPLAY_NAMES, display_flat):
        model_data = combined_df[combined_df['Model'] == display_name]
        for cond in condition_order:
            scores = model_data[model_data['Condition'] == cond]['Score']
            if scores.std() == 0:
                val = scores.iloc[0]
                ax.axvline(val, color=palette[cond], linewidth=3, label=cond)
                ax.fill_betweenx([0, ax.get_ylim()[1] or 10], val - 0.01, val + 0.01,
                                 color=palette[cond], alpha=0.3)
            else:
                sns.kdeplot(scores, label=cond, color=palette[cond], fill=True, alpha=0.3,
                            linewidth=2, clip=(0, 1), ax=ax)
        ax.set_xlim(0, 1)
        ax.set_title(display_flat_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('StrongREJECT Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=10)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(f"{output_dir}/kde_{file_tag}.{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_dir}/kde_{file_tag}.png/pdf")
    plt.close()


def make_poster_plot(zero_pad=False):
    """Generate poster plots for CoT intervention."""
    common_refused, before_dfs = get_common_refused_prompts()
    expected_per_model = len(common_refused) * N_REPS

    cot_layers = {}
    after_dfs = {}
    for model_name in MODEL_NAMES:
        with open(f"results/{model_name}/.best_layer_cot") as f:
            cot_layers[model_name] = int(f.read().strip())
        after_path = f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layers[model_name]}.csv"
        after_dfs[model_name] = pd.read_csv(after_path)

    all_data = []
    for model_name, display_name in zip(MODEL_NAMES, DISPLAY_NAMES):
        before_filtered = before_dfs[model_name][before_dfs[model_name]['prompt'].isin(common_refused)]
        after_filtered = after_dfs[model_name][after_dfs[model_name]['prompt'].isin(common_refused)]

        before_scores = before_filtered['strongreject_score'].dropna().clip(lower=0, upper=1)
        after_scores = after_filtered['strongreject_score'].dropna().clip(lower=0, upper=1)

        if zero_pad:
            before_scores = pad_scores_with_zeros(before_scores, expected_per_model)
            after_scores = pad_scores_with_zeros(after_scores, expected_per_model)

        for score in before_scores:
            all_data.append({'Score': score, 'Condition': 'Normal', 'Model': display_name})
        for score in after_scores:
            all_data.append({'Score': score, 'Condition': 'CoT Intervention', 'Model': display_name})

        print(f"{display_name.replace(chr(10), ' ')} (CoT layer {cot_layers[model_name]}): "
              f"before n={len(before_scores)}, after n={len(after_scores)}, "
              f"after mean={after_scores.mean():.3f}")

    combined_df = pd.DataFrame(all_data)
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    condition_order = ['Normal', 'CoT Intervention']
    palette = {'Normal': "#569e56", 'CoT Intervention': "#d44949"}

    file_tag = 'poster_zeropadded' if zero_pad else 'poster'
    _render_poster_plots(combined_df, condition_order, palette, output_dir, file_tag)


def make_poster_plot_baseline(zero_pad=False):
    """Generate poster plots for end-of-prompt intervention, using the same prompt subset as CoT."""
    common_refused, before_dfs = get_common_refused_prompts()
    expected_per_model = len(common_refused) * N_REPS

    baseline_layers = {}
    after_dfs = {}
    for model_name in MODEL_NAMES:
        with open(f"results/{model_name}/.best_layer_baseline") as f:
            baseline_layers[model_name] = int(f.read().strip())
        after_path = f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layers[model_name]}.csv"
        after_dfs[model_name] = pd.read_csv(after_path)

    all_data = []
    for model_name, display_name in zip(MODEL_NAMES, DISPLAY_NAMES):
        before_filtered = before_dfs[model_name][before_dfs[model_name]['prompt'].isin(common_refused)]
        after_filtered = after_dfs[model_name][after_dfs[model_name]['prompt'].isin(common_refused)]

        before_scores = before_filtered['strongreject_score'].dropna().clip(lower=0, upper=1)
        after_scores = after_filtered['strongreject_score'].dropna().clip(lower=0, upper=1)

        if zero_pad:
            before_scores = pad_scores_with_zeros(before_scores, expected_per_model)
            after_scores = pad_scores_with_zeros(after_scores, expected_per_model)

        for score in before_scores:
            all_data.append({'Score': score, 'Condition': 'Normal', 'Model': display_name})
        for score in after_scores:
            all_data.append({'Score': score, 'Condition': 'Prompt Intervention', 'Model': display_name})

        print(f"{display_name.replace(chr(10), ' ')} (end-of-prompt layer {baseline_layers[model_name]}): "
              f"before n={len(before_scores)}, after n={len(after_scores)}, "
              f"after mean={after_scores.mean():.3f}")

    combined_df = pd.DataFrame(all_data)
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    condition_order = ['Normal', 'Prompt Intervention']
    palette = {'Normal': "#569e56", 'Prompt Intervention': "#d44949"}

    file_tag = 'poster_end_of_prompt_zeropadded' if zero_pad else 'poster_end_of_prompt'
    _render_poster_plots(combined_df, condition_order, palette, output_dir, file_tag)


def make_poster_plot_both(zero_pad=False):
    """Generate poster plots with Before, After (CoT), and After (prompt) on the same plot."""
    common_refused, before_dfs = get_common_refused_prompts()
    expected_per_model = len(common_refused) * N_REPS

    cot_layers = {}
    baseline_layers = {}
    cot_dfs = {}
    baseline_dfs = {}
    for model_name in MODEL_NAMES:
        with open(f"results/{model_name}/.best_layer_cot") as f:
            cot_layers[model_name] = int(f.read().strip())
        with open(f"results/{model_name}/.best_layer_baseline") as f:
            baseline_layers[model_name] = int(f.read().strip())
        cot_dfs[model_name] = pd.read_csv(
            f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layers[model_name]}.csv")
        baseline_dfs[model_name] = pd.read_csv(
            f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layers[model_name]}.csv")

    all_data = []
    for model_name, display_name in zip(MODEL_NAMES, DISPLAY_NAMES):
        before_filtered = before_dfs[model_name][before_dfs[model_name]['prompt'].isin(common_refused)]
        cot_filtered = cot_dfs[model_name][cot_dfs[model_name]['prompt'].isin(common_refused)]
        bl_filtered = baseline_dfs[model_name][baseline_dfs[model_name]['prompt'].isin(common_refused)]

        before_scores = before_filtered['strongreject_score'].dropna().clip(lower=0, upper=1)
        cot_scores = cot_filtered['strongreject_score'].dropna().clip(lower=0, upper=1)
        bl_scores = bl_filtered['strongreject_score'].dropna().clip(lower=0, upper=1)

        if zero_pad:
            before_scores = pad_scores_with_zeros(before_scores, expected_per_model)
            cot_scores = pad_scores_with_zeros(cot_scores, expected_per_model)
            bl_scores = pad_scores_with_zeros(bl_scores, expected_per_model)

        for score in before_scores:
            all_data.append({'Score': score, 'Condition': 'Normal', 'Model': display_name})
        for score in cot_scores:
            all_data.append({'Score': score, 'Condition': 'CoT Intervention', 'Model': display_name})
        for score in bl_scores:
            all_data.append({'Score': score, 'Condition': 'Prompt Intervention', 'Model': display_name})

        print(f"{display_name.replace(chr(10), ' ')} "
              f"(CoT layer {cot_layers[model_name]}, prompt layer {baseline_layers[model_name]}): "
              f"before n={len(before_scores)}, "
              f"after(CoT) n={len(cot_scores)} mean={cot_scores.mean():.3f}, "
              f"after(prompt) n={len(bl_scores)} mean={bl_scores.mean():.3f}")

    combined_df = pd.DataFrame(all_data)
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    condition_order = ['Normal', 'Prompt Intervention', 'CoT Intervention']
    palette = {'Normal': "#009c1f", 'Prompt Intervention': "#6666FF", 'CoT Intervention': "#FF9933"}

    file_tag = 'poster_both_zeropadded' if zero_pad else 'poster_both'
    _render_poster_plots(combined_df, condition_order, palette, output_dir, file_tag)


def make_poster_plot_both_full(zero_pad=False):
    """Generate poster plots with all three conditions on the full holdout set (no prompt filtering)."""
    cot_layers = {}
    baseline_layers = {}
    before_dfs = {}
    cot_dfs = {}
    baseline_dfs = {}
    for model_name in MODEL_NAMES:
        before_dfs[model_name] = pd.read_csv(
            f"results/{model_name}/dataset/scored_test_harmful_prompts_cot5_out5.csv")
        with open(f"results/{model_name}/.best_layer_cot") as f:
            cot_layers[model_name] = int(f.read().strip())
        with open(f"results/{model_name}/.best_layer_baseline") as f:
            baseline_layers[model_name] = int(f.read().strip())
        cot_dfs[model_name] = pd.read_csv(
            f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layers[model_name]}.csv")
        baseline_dfs[model_name] = pd.read_csv(
            f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layers[model_name]}.csv")

    all_data = []
    for model_name, display_name in zip(MODEL_NAMES, DISPLAY_NAMES):
        n_prompts = before_dfs[model_name]['prompt'].nunique()
        expected = n_prompts * N_REPS

        before_scores = before_dfs[model_name]['strongreject_score'].dropna().clip(lower=0, upper=1)
        cot_scores = cot_dfs[model_name]['strongreject_score'].dropna().clip(lower=0, upper=1)
        bl_scores = baseline_dfs[model_name]['strongreject_score'].dropna().clip(lower=0, upper=1)

        if zero_pad:
            before_scores = pad_scores_with_zeros(before_scores, expected)
            cot_scores = pad_scores_with_zeros(cot_scores, expected)
            bl_scores = pad_scores_with_zeros(bl_scores, expected)

        for score in before_scores:
            all_data.append({'Score': score, 'Condition': 'Normal', 'Model': display_name})
        for score in cot_scores:
            all_data.append({'Score': score, 'Condition': 'CoT Intervention', 'Model': display_name})
        for score in bl_scores:
            all_data.append({'Score': score, 'Condition': 'Prompt Intervention', 'Model': display_name})

        print(f"{display_name.replace(chr(10), ' ')} "
              f"(CoT layer {cot_layers[model_name]}, prompt layer {baseline_layers[model_name]}): "
              f"before n={len(before_scores)}, "
              f"after(CoT) n={len(cot_scores)} mean={cot_scores.mean():.3f}, "
              f"after(prompt) n={len(bl_scores)} mean={bl_scores.mean():.3f}")

    combined_df = pd.DataFrame(all_data)
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    condition_order = ['Normal', 'Prompt Intervention', 'CoT Intervention']
    palette = {'Normal': "#009c1f", 'Prompt Intervention': "#6666FF", 'CoT Intervention': "#FF9933"}

    file_tag = 'poster_both_full_zeropadded' if zero_pad else 'poster_both_full'
    _render_poster_plots(combined_df, condition_order, palette, output_dir, file_tag)


def main():
    args = parse_args()

    if args.poster:
        make_poster_plot(zero_pad=args.zero_pad)
        return

    if args.poster_baseline:
        make_poster_plot_baseline(zero_pad=args.zero_pad)
        return

    if args.poster_both:
        make_poster_plot_both(zero_pad=args.zero_pad)
        return

    if args.poster_both_full:
        make_poster_plot_both_full(zero_pad=args.zero_pad)
        return

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

"""
Generate plots comparing StrongReject scores before and after intervention across all models.
Failed generations (dropped by the pipeline) are imputed as score=0 to reflect true success rates.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def pad_scores_with_zeros(scores, max_rows):
    """Pad a scores series with zeros to account for failed generations."""
    n_missing = max_rows - len(scores)
    if n_missing > 0:
        zeros = pd.Series(np.zeros(n_missing))
        return pd.concat([scores, zeros], ignore_index=True)
    return scores


CONDITION_ORDER = [
    'Before\nintervention',
    'Harmful dir.,\nbaseline acts.',
    'Harmful dir.,\nCoT acts.',
    'Harmless dir.,\nbaseline acts.',
    'Harmless dir.,\nCoT acts.',
]

PALETTE = {
    'Before\nintervention': '#999999',
    'Harmful dir.,\nbaseline acts.': '#7bafd4',   # light blue (darkened for readability)
    'Harmful dir.,\nCoT acts.': '#1f78b4',         # dark blue
    'Harmless dir.,\nbaseline acts.': '#fdbf6f',   # light orange
    'Harmless dir.,\nCoT acts.': '#e68a00',         # dark orange
}


def style_ax(ax, title, display_names, layer_lists):
    """Apply shared styling: title, labels, legend, layer annotations, comply/refuse markers."""
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Model & Layer Intervention', fontsize=12)
    ax.set_ylabel('StrongReject Score', fontsize=12)

    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks([])
    ax_right.set_ylabel('')
    ax.annotate('(comply)', xy=(1.01, 1), xycoords='axes fraction', fontsize=10, fontstyle='italic', va='top')
    ax.annotate('(refuse)', xy=(1.01, 0), xycoords='axes fraction', fontsize=10, fontstyle='italic', va='bottom')
    ax.legend(title='Condition', bbox_to_anchor=(1.02, 0.92), loc='upper left')

    # Color-coded layer numbers below each bar/violin
    n_hue = len(CONDITION_ORDER)
    box_width = 0.8 / n_hue
    offsets = [(i - (n_hue - 1) / 2) * box_width for i in range(n_hue)]
    colors = [PALETTE[c] for c in CONDITION_ORDER]
    trans = ax.get_xaxis_transform()
    y_layer = -0.05

    for model_idx in range(len(display_names)):
        for hue_idx in range(n_hue):
            x = model_idx + offsets[hue_idx]
            if hue_idx == 0:
                label, color = '—', colors[0]
            else:
                label, color = str(layer_lists[hue_idx][model_idx]), colors[hue_idx]
            ax.text(x, y_layer, label, ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color,
                    transform=trans, clip_on=False)

    ax.tick_params(axis='x', pad=22)

    # Add minor ticks at each hue position (above the layer numbers)
    minor_ticks = []
    for model_idx in range(len(display_names)):
        for hue_idx in range(n_hue):
            minor_ticks.append(model_idx + offsets[hue_idx])
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=4, width=0.8, direction='out')


def main():
    MODEL1 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    MODEL2 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MODEL3 = "Qwen/Qwen3-8B"
    MODEL4 = "openai/gpt-oss-20b"

    model_names = [MODEL1, MODEL2, MODEL3, MODEL4]
    display_names = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-7B", "Qwen3-8B", "GPT-OSS-20B"]

    cot_layers = [17, 17, 13, 19]
    baseline_layers = [11, 17, 21, 15]
    cot_harmless_layers = [17, 23, 19, 7]
    baseline_harmless_layers = [13, 15, 19, 17]

    layer_lists = {
        1: baseline_layers,
        2: cot_layers,
        3: baseline_harmless_layers,
        4: cot_harmless_layers,
    }

    MAX_ROWS = 487 * 5 * 5  # 12175


    all_data = []
    conditions = [
        ('before', 'Before\nintervention'),
        ('baseline', 'Harmful dir.,\nbaseline acts.'),
        ('cot', 'Harmful dir.,\nCoT acts.'),
        ('baseline_harmless', 'Harmless dir.,\nbaseline acts.'),
        ('cot_harmless', 'Harmless dir.,\nCoT acts.'),
    ]

    for model_name, display_name, cot_layer, baseline_layer, cot_harmless_layer, baseline_harmless_layer in zip(
        model_names, display_names, cot_layers, baseline_layers, cot_harmless_layers, baseline_harmless_layers
    ):
        paths = {
            'before': f"results/{model_name}/dataset/scored_test_harmful_prompts_cot5_out5.csv",
            'cot': f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}.csv",
            'baseline': f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layer}.csv",
            'cot_harmless': f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_harmless_layer}_harmless.csv",
            'baseline_harmless': f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_harmless_layer}_harmless.csv",
        }

        counts = {}
        for key, condition_label in conditions:
            df = pd.read_csv(paths[key])
            raw_scores = df['strongreject_score'].dropna()
            counts[key] = len(raw_scores)
            scores = pad_scores_with_zeros(raw_scores, MAX_ROWS)
            for score in scores:
                all_data.append({'Score': score, 'Condition': condition_label, 'Model': display_name})

        print(f"{display_name}: " + ", ".join(
            f"{k}={counts[k]}/{MAX_ROWS} ({counts[k]/MAX_ROWS*100:.1f}%)" for k, _ in conditions
        ))

    combined_df = pd.DataFrame(all_data)
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Box plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=combined_df, x='Model', y='Score', hue='Condition',
                hue_order=CONDITION_ORDER, palette=PALETTE, showfliers=False, ax=ax)
    style_ax(ax, 'StrongReject score comparison across models, before and after weight orthogonalisation',
             display_names, layer_lists)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_all_models_both_datasets.png", dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {output_dir}/boxplot_all_models_both_datasets.png")

    # Violin plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(data=combined_df, x='Model', y='Score', hue='Condition',
                   hue_order=CONDITION_ORDER, palette=PALETTE,
                   cut=0, inner='quartile', density_norm='width', gap=0.15, ax=ax)
    style_ax(ax, 'StrongReject score distribution across models, before and after weight orthogonalisation',
             display_names, layer_lists)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/violin_all_models_both_datasets.png", dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to: {output_dir}/violin_all_models_both_datasets.png")


if __name__ == "__main__":
    main()

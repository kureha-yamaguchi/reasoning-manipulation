"""
Simplified box plot for poster: Before vs After (CoT, harmful dir.) across all 4 models.
Reads best layers from .best_layer_cot dot files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def pad_scores_with_zeros(scores, max_rows):
    n_missing = max_rows - len(scores)
    if n_missing > 0:
        zeros = pd.Series(np.zeros(n_missing))
        return pd.concat([scores, zeros], ignore_index=True)
    return scores


CONDITION_ORDER = ['Before', 'After']
PALETTE = {'Before': '#d62728', 'After': '#2ca02c'}


def main():
    model_names = [
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Qwen/Qwen3-8B",
        "openai/gpt-oss-20b",
    ]
    display_names = [
        "DeepSeek-R1-\nDistill-Llama-8B",
        "DeepSeek-R1-\nDistill-Qwen-7B",
        "Qwen3-8B",
        "GPT-OSS-20B",
    ]

    N_REPS = 5 * 5  # cot_reps * output_reps

    all_data = []
    for model_name, display_name in zip(model_names, display_names):
        # Read best CoT layer from dot file
        layer_file = f"results/{model_name}/.best_layer_cot"
        with open(layer_file) as f:
            cot_layer = int(f.read().strip())

        before_path = f"results/{model_name}/dataset/scored_test_harmful_prompts_cot5_out5.csv"
        after_path = f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}.csv"

        before_df = pd.read_csv(before_path)
        after_df = pd.read_csv(after_path)

        # Max rows = number of unique prompts (union of both) * reps
        n_prompts = max(before_df['prompt'].nunique(), after_df['prompt'].nunique())
        max_rows = n_prompts * N_REPS

        for df, label in [(before_df, 'Before'), (after_df, 'After')]:
            raw_scores = df['strongreject_score'].dropna().clip(0, 1)
            scores = pad_scores_with_zeros(raw_scores, max_rows)
            for score in scores:
                all_data.append({'Score': score, 'Condition': label, 'Model': display_name})

        print(f"{display_name.replace(chr(10), ' ')} (layer {cot_layer}): "
              f"before={len(before_df)}/{max_rows}, after={len(after_df)}/{max_rows}")

    combined_df = pd.DataFrame(all_data)
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=combined_df, x='Model', y='Score', hue='Condition',
                hue_order=CONDITION_ORDER, palette=PALETTE, showfliers=False,
                width=0.6, linewidth=1.5, ax=ax)

    ax.set_title('StrongREJECT scores before and after\nweight orthogonalisation', fontsize=16, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('StrongREJECT Score', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title='', fontsize=13, loc='upper right')

    ax_right = ax.secondary_yaxis('right')
    ax_right.set_yticks([])
    ax_right.set_ylabel('')
    ax.annotate('(comply)', xy=(1.01, 1), xycoords='axes fraction', fontsize=11, fontstyle='italic', va='top')
    ax.annotate('(refuse)', xy=(1.01, 0), xycoords='axes fraction', fontsize=11, fontstyle='italic', va='bottom')

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = f"{output_dir}/boxplot_poster.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

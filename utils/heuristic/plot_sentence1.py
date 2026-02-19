import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
model_name = 'Qwen/Qwen3-8B'
index_number = 2
cot_numbers = [1, 2, 3, 4, 5]
output_target = [0.72, 0.00, 0.84, 0.01, 0.83]
scores_all = []

for i in cot_numbers:
    input_csv = f'scored_resampling_results_idx{index_number}_cot{i}.csv'
    input_path = os.path.join('results', model_name, 'dataset', input_csv)
    df = pd.read_csv(input_path)
    scores_all.append(df['strongreject_score'].values)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Simple box plot
bp = ax.boxplot(scores_all, positions=cot_numbers, widths=0.6,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='black'))

# Red markers for output_target (no line)
ax.scatter(cot_numbers, output_target, c='red', s=80, marker='s', 
           label='Output Target', zorder=5)

ax.set_xlabel('CoT Number', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Output Target vs Resampling after the First CoT Sentence (25 rollouts)', fontsize=14)
ax.set_xticks(cot_numbers)
ax.legend([bp["boxes"][0], ax.collections[0]], ['Resampling Distribution', 'Output Target'])
ax.grid(True, alpha=0.3)
ax.annotate('(comply)', xy=(1.01, 1), xycoords='axes fraction', fontsize=10, fontstyle='italic', va='top')
ax.annotate('(refuse)', xy=(1.01, 0), xycoords='axes fraction', fontsize=10, fontstyle='italic', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join('results', model_name, 'figures', f'cot_scores_plot_index_{index_number}.png'), dpi=150)
plt.show()
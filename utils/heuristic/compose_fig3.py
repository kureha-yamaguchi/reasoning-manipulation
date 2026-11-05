'''
Fig 3 for several models as one figure (not tiled single-model PDFs): per model, the within-prefix SD
curve over the top row and the 3-prompt heatmap below, with shared axes. Axis labels appear once
(y-labels in the first column, one x-label per row) and there is a single colourbar. Each model's
heatmap prompts and seed are read from its single-model panel's _prompts.json, so the rows match.

uv run -m utils.heuristic.compose_fig3 --model_names deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B QuixiAI/DeepSeek-R1-AWQ openai/gpt-oss-20b
'''

import argparse
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt

from utils.heuristic.plot_quadrant_matrix import (
    draw_colorbar,
    draw_heatmap,
    draw_std,
    eligible_paths,
    model_title,
    select_data,
)

SHORT = {'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': 'R1-Distill-Llama-8B',
         'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': 'R1-Distill-Qwen-7B',
         'nvidia/AceReason-Nemotron-7B': 'AceReason-Nemotron-7B', 'QuixiAI/DeepSeek-R1-AWQ': 'DeepSeek-R1 (AWQ)',
         'openai/gpt-oss-20b': 'gpt-oss-20b', 'openai/gpt-oss-120b': 'gpt-oss-120b'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_names', nargs='+', required=True)
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--n_bins', type=int, default=30)
    parser.add_argument('--out', default='figures/fig3_quadrant_panels.pdf')
    args = parser.parse_args()

    data = []
    for m in args.model_names:
        with open(os.path.join(args.results_dir, m, 'figures',
                               f'heatmap_quadrant_rep{args.repetitions}_bins{args.n_bins}_sampled_prompts.json')) as f:
            pinned = json.load(f)
        a = SimpleNamespace(seed=pinned['seed'], random_sample=True, n_bins=args.n_bins,
                            heatmap_prompts=','.join(map(str, pinned['heatmap_prompts'])))
        data.append(select_data(eligible_paths(m, args.results_dir, args.repetitions), a))

    n, fs = len(args.model_names), 22
    heat_h = max(3, 0.35 * max(d['mean_mat'].shape[0] for d in data))
    fig = plt.figure(figsize=(6.6 * n + 1.6, heat_h + 3.6))
    gs = fig.add_gridspec(2, n + 1, height_ratios=[3.0, heat_h], width_ratios=[1] * n + [0.05],
                          hspace=0.12, wspace=0.34)
    line_axes, im = [], None
    for j, (m, d) in enumerate(zip(args.model_names, data)):
        ax_line = fig.add_subplot(gs[0, j], sharey=line_axes[0] if line_axes else None)
        draw_std(ax_line, d, args.n_bins, ylabel=(j == 0), xlabel=False, fs=fs)
        ax_line.set_title(SHORT.get(m, model_title(m)), fontsize=fs, fontweight='bold', pad=12)
        ax_line.tick_params(labelbottom=False)
        if j > 0:
            ax_line.tick_params(labelleft=False)
        line_axes.append(ax_line)
        im = draw_heatmap(fig.add_subplot(gs[1, j]), d, args.n_bins, ylabel=(j == 0), xlabel=False, fs=fs)
    draw_colorbar(fig, im, fig.add_subplot(gs[1, n]), fs=fs)
    fig.supxlabel('Normalised CoT sentence position', fontsize=fs, y=0.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(os.path.splitext(args.out)[0] + '.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {args.out} (+ .png): {", ".join(SHORT.get(m, model_title(m)) for m in args.model_names)}')


if __name__ == '__main__':
    main()

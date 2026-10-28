'''
Overlay, for several models, the across-rollout std of StrongREJECT scores vs normalised
CoT sentence position (the top panel of paper Fig 3), on shared axes.

Binning matches plot_quadrant_matrix.build_matrices. Unlike Fig 3 (mean over (prompt, CoT)
rows, band = ±1 SD), curves here average CoTs within each prompt first, then show the mean
over prompts with a 95% cluster-bootstrap CI over prompts.

uv run -m utils.heuristic.plot_std_overlay --model_names \
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B openai/gpt-oss-20b
'''

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.heuristic.plot_quadrant_matrix import build_matrices, extract_prompt_cot

# Okabe-Ito (colourblind-safe); fixed per model so colours are consistent across figures
MODEL_COLOURS = {
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': '#0072B2',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': '#009E73',
    'openai/gpt-oss-20b': '#D55E00',
    'Qwen/Qwen3-8B': '#CC79A7',
}


def prompt_curves(model_name: str, results_dir: str, repetitions: int, n_bins: int):
    """(n_prompts × n_bins) std curves, CoTs averaged within prompt."""
    paths = sorted(glob.glob(os.path.join(results_dir, model_name, 'dataset', 'quadrants',
                                          f'scored_full_resample_prompt*_cot*_rep_{repetitions}.csv')))
    _, std_mat = build_matrices(paths, n_bins)
    prompts = np.array([extract_prompt_cot(p)[0] for p in paths])
    return np.stack([np.nanmean(std_mat[prompts == p], axis=0) for p in np.unique(prompts)])


def bootstrap_band(curves: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(curves), size=(n_boot, len(curves)))
    boots = np.nanmean(curves[idx], axis=1)
    return np.nanquantile(boots, alpha / 2, axis=0), np.nanquantile(boots, 1 - alpha / 2, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_names', nargs='+', required=True)
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--n_bins', type=int, default=30)
    parser.add_argument('--n_boot', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='figures/std_overlay_rep10_bins30.pdf')
    args = parser.parse_args()

    x = np.linspace(0.0, 1.0, args.n_bins)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cache = {}
    for model in args.model_names:
        curves = prompt_curves(model, args.results_dir, args.repetitions, args.n_bins)
        lo, hi = bootstrap_band(curves, args.n_boot, args.seed)
        mean = np.nanmean(curves, axis=0)
        colour = MODEL_COLOURS.get(model)
        ax.plot(x, mean, color=colour, linewidth=2, label=f'{model.split("/")[-1]} (N={len(curves)})')
        ax.fill_between(x, lo, hi, color=colour, alpha=0.2, linewidth=0)
        cache[model] = {'mean': mean.tolist(), 'ci95_lo': lo.tolist(), 'ci95_hi': hi.tolist(), 'n_prompts': len(curves)}

    ax.set_xlabel('Normalised CoT sentence position')
    ax.set_ylabel('Std of StrongREJECT score\nacross k=10 rollouts')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(os.path.splitext(args.out)[0] + '.png', dpi=200, bbox_inches='tight')
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump({'args': vars(args), 'x': x.tolist(), 'models': cache}, f)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()

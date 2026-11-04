'''
Plot heatmap of mean StrongReject score over prompt × normalised CoT sentence position,
plus line graphs of aggregated mean and standard deviation across prompts.

Each row is one (prompt, cot) pair from a CSV file.
Rows are clustered by prompt index, with horizontal borders separating groups
and square brackets on the y-axis labelling each cluster with its prompt index.

uv run -m utils.heuristic.plot_quadrant_matrix --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --no_colorbar
uv run -m utils.heuristic.plot_quadrant_matrix --model_name openai/gpt-oss-20b

Paper Fig 3 heatmaps (prompts pinned; line-plot subsample fixed by --seed):
uv run -m utils.heuristic.plot_quadrant_matrix --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --no_colorbar --random_sample --heatmap_prompts 806,963,1073
uv run -m utils.heuristic.plot_quadrant_matrix --model_name openai/gpt-oss-20b --random_sample --heatmap_prompts 571,611,2057
'''

import argparse
import glob
import json
import os
import re
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot heatmap of StrongReject score vs normalised sentence position"
    )
    parser.add_argument("--model_name", type=str,
                        default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                        help="Model name (used to locate result CSVs)")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--repetitions", type=int, default=10,
                        help="Number of output variations per prompt + CoT prefix")
    parser.add_argument("--n_bins", type=int, default=30,
                        help="Number of bins for normalised sentence position")
    parser.add_argument("--no_colorbar", action="store_true",
                        help="Omit the colorbar from the figure")
    parser.add_argument("--all_files", action="store_true",
                        help="Use every rollout file, not only prompts in the latest run's eligible set")
    parser.add_argument("--random_sample", action="store_true",
                        help="Trim heatmap to ~3 random examples per model")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for prompt subsampling (line plot and heatmap)")
    parser.add_argument("--heatmap_prompts", type=str, default=None,
                        help="Comma-separated prompt indices for the heatmap, overriding "
                             "random selection (e.g. paper Fig 3: 806,963,1073 / 571,611,2057)")
    return parser.parse_args()


def extract_prompt_cot(filepath: str) -> tuple[int, int]:
    """Extract prompt index and cot index from a filename like
    scored_full_resample_prompt200_cot3_rep_10.csv"""
    basename = os.path.basename(filepath)
    m = re.search(r'prompt(\d+)_cot(\d+)', basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse prompt/cot from filename: {basename}")


def build_matrices(input_paths: list[str],
                   n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (num_files × n_bins) matrices of per-sentence mean and std scores.
    """
    bin_centres = np.linspace(0.0, 1.0, n_bins)

    mean_matrix = np.full((len(input_paths), n_bins), np.nan)
    std_matrix = np.full((len(input_paths), n_bins), np.nan)

    for row_idx, input_path in enumerate(input_paths):
        df = pd.read_csv(input_path)

        max_idx = df['sentence_idx'].max()
        if max_idx == 0:
            mean_val = df['strongreject_score'].mean()
            std_val = df['strongreject_score'].std()
            mean_matrix[row_idx, :] = mean_val
            std_matrix[row_idx, :] = std_val if not np.isnan(std_val) else 0.0
            continue

        df['norm_pos'] = df['sentence_idx'] / max_idx

        mean_at_pos = df.groupby('norm_pos')['strongreject_score'].mean().sort_index()
        mean_at_pos = mean_at_pos.dropna()
        std_at_pos = df.groupby('norm_pos')['strongreject_score'].std().fillna(0).sort_index()
        std_at_pos = std_at_pos.loc[mean_at_pos.index]

        mean_matrix[row_idx, :] = np.interp(
            bin_centres, mean_at_pos.index.values, mean_at_pos.values
        )
        std_matrix[row_idx, :] = np.interp(
            bin_centres, std_at_pos.index.values, std_at_pos.values
        )

    return mean_matrix, std_matrix


def select_data(input_paths: list[str], args) -> dict:
    """Prompts for the line plot (up to 32) and the heatmap (3, or args.heatmap_prompts), their binned
    matrices and the heatmap's prompt-cluster boundaries."""
    file_info = sorted(({'path': p, 'prompt_idx': extract_prompt_cot(p)[0], 'cot_idx': extract_prompt_cot(p)[1]}
                        for p in input_paths), key=lambda x: (x['prompt_idx'], x['cot_idx']))
    all_prompts = list(dict.fromkeys(f['prompt_idx'] for f in file_info))
    rng = np.random.default_rng(args.seed)
    if args.random_sample:
        h = 0.3
        sampled_line = set(rng.choice(all_prompts, size=min(32, len(all_prompts)), replace=False).tolist())
        if args.heatmap_prompts:  # explicit prompts, or 3 random prompts each with exactly 5 CoTs
            sampled_heat = {int(p) for p in args.heatmap_prompts.split(',')}
            missing = sampled_heat - set(all_prompts)
            if missing:
                raise ValueError(f"Heatmap prompts not found in results: {sorted(missing)}")
        else:
            cot_counts = Counter(f['prompt_idx'] for f in file_info)
            prompts_with_5_cots = [p for p, c in cot_counts.items() if c == 5]
            sampled_heat = set(rng.choice(prompts_with_5_cots, size=min(3, len(prompts_with_5_cots)),
                                          replace=False).tolist())
    else:
        h = 0.05
        sampled_line = sampled_heat = set(rng.choice(all_prompts, size=min(32, len(all_prompts)),
                                                     replace=False).tolist())
    file_info_heat = [f for f in file_info if f['prompt_idx'] in sampled_heat]
    file_info_line = [f for f in file_info if f['prompt_idx'] in sampled_line]
    mean_mat, _ = build_matrices([f['path'] for f in file_info_heat], args.n_bins)
    _, std_mat = build_matrices([f['path'] for f in file_info_line], args.n_bins)
    order = [f['prompt_idx'] for f in file_info_heat]
    unique_prompts, bounds = [], []
    for i, pi in enumerate(order):
        if i == 0 or pi != order[i - 1]:
            unique_prompts.append(pi)
            bounds.append(i)
    bounds.append(len(file_info_heat))
    return {'mean_mat': mean_mat, 'std_mat': std_mat, 'unique_prompts': unique_prompts, 'bounds': bounds,
            'sampled_line': sorted(sampled_line), 'sampled_heat': sorted(sampled_heat), 'hspace': h}


def draw_heatmap(ax, d: dict, n_bins: int, ylabel: bool = True, xlabel: bool = True, fs: float = 22):
    """Per-CoT mean StrongREJECT score by normalised sentence position, rows bracketed by prompt."""
    im = ax.imshow(d['mean_mat'], aspect='auto', cmap='Reds', vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_xticks(np.linspace(0, n_bins - 1, 6))
    ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(0, 1, 6)], fontsize=fs)
    if xlabel:
        ax.set_xlabel('Normalised CoT Sentence Position', fontsize=fs)
    ax.set_yticks([])
    if ylabel:
        ax.set_ylabel('Prompt Index', fontsize=fs, labelpad=65)
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    gap, bounds = 0.15, d['bounds']
    for g, pi in enumerate(d['unique_prompts']):
        start, end = bounds[g] - 0.5 + gap, bounds[g + 1] - 0.5 - gap
        ax.plot([-0.028, -0.012, -0.012, -0.028], [start, start, end, end], transform=trans, color='black',
                linewidth=1.5, clip_on=False, solid_capstyle='butt')
        ax.text(-0.055, (start + end) / 2, str(pi), transform=trans, ha='right', va='center', fontsize=fs - 4)
    for boundary in bounds[1:-1]:  # white separators between prompt clusters
        ax.axhline(y=boundary - 0.5, color='white', linewidth=5.0)
    for spine in ax.spines.values():  # black border around each cluster, no continuous outer spine
        spine.set_visible(False)
    n = len(d['unique_prompts'])
    for g in range(n):
        y0 = bounds[g] - 0.5 + (gap if g > 0 else 0)
        y1 = bounds[g + 1] - 0.5 - (gap if g < n - 1 else 0)
        ax.add_patch(mpatches.Rectangle((-0.5, y0), n_bins, y1 - y0, linewidth=1.5, edgecolor='black',
                                        facecolor='none'))
    return im


def draw_std(ax, d: dict, n_bins: int, ylabel: bool = True, xlabel: bool = True, fs: float = 22):
    """Within-prefix SD across rollouts, mean over (prompt, CoT) rows with a +/- 1 SD band."""
    x = np.linspace(0.0, 1.0, n_bins)
    avg, sd = np.nanmean(d['std_mat'], axis=0), np.nanstd(d['std_mat'], axis=0)
    ax.plot(x, avg, linewidth=2, color='#2166ac')
    ax.fill_between(x, avg - sd, avg + sd, alpha=0.25, color='#2166ac')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    if xlabel:
        ax.set_xlabel('Normalised CoT Sentence Position', fontsize=fs)
    if ylabel:
        ax.set_ylabel('Mean Std Dev\n(across rollouts)', fontsize=fs - 2)
    ax.tick_params(labelsize=fs)
    ax.grid(True, alpha=0.3)


def draw_colorbar(fig, im, cax, fs: float = 22):
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Mean StrongREJECT Score', fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.annotate('comply', xy=(0.5, 1.08), xycoords='axes fraction', fontsize=fs - 4, fontstyle='italic',
                     ha='center')
    cbar.ax.annotate('refuse', xy=(0.5, -0.08), xycoords='axes fraction', fontsize=fs - 4, fontstyle='italic',
                     ha='center')


def model_title(model_name: str) -> str:
    return 'GPT-OSS-20B' if model_name == 'openai/gpt-oss-20b' else model_name.split('/')[-1]


def plot_matrix(input_paths: list[str], args):
    if not input_paths:
        print("No data to plot.")
        return
    d = select_data(input_paths, args)
    args.sampled_line, args.sampled_heat = d['sampled_line'], d['sampled_heat']

    heatmap_height, line_height, panel_width, cbar_width = max(3, 0.35 * d['mean_mat'].shape[0]), 3.0, 7, 0.4
    fig = plt.figure(figsize=(panel_width + (0 if args.no_colorbar else cbar_width + 1.5), heatmap_height + line_height))
    fig.suptitle(model_title(args.model_name), fontsize=22, fontweight='bold', y=1.07)
    fig.subplots_adjust(top=1.0)
    if args.no_colorbar:
        gs = fig.add_gridspec(2, 1, height_ratios=[line_height, heatmap_height], hspace=d['hspace'])
    else:
        gs = fig.add_gridspec(2, 2, height_ratios=[line_height, heatmap_height],
                              width_ratios=[1, cbar_width / panel_width], hspace=d['hspace'], wspace=0.2)
    im = draw_heatmap(fig.add_subplot(gs[1, 0]), d, args.n_bins, ylabel=args.no_colorbar)
    if not args.no_colorbar:
        draw_colorbar(fig, im, fig.add_subplot(gs[1, 1]))
    draw_std(fig.add_subplot(gs[0, 0]), d, args.n_bins, ylabel=args.no_colorbar)

    out_dir = os.path.join(args.results_dir, args.model_name, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    suffix = '_sampled' if args.random_sample else ''
    output_path = os.path.join(out_dir, f'heatmap_quadrant_rep{args.repetitions}_bins{args.n_bins}{suffix}.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    with open(os.path.splitext(output_path)[0] + '_prompts.json', 'w') as f:  # to reproduce the figure
        json.dump({'seed': args.seed, 'line_prompts': args.sampled_line, 'heatmap_prompts': args.sampled_heat}, f,
                  indent=2)
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def eligible_paths(model_name: str, results_dir: str, repetitions: int, all_files: bool = False) -> list[str]:
    paths = sorted(glob.glob(os.path.join(results_dir, model_name, 'dataset', 'quadrants',
                                          f'scored_full_resample_prompt*_cot*_rep_{repetitions}.csv')))
    if all_files:
        return paths
    from utils.heuristic.commitment import scored_paths  # same eligible-prompt restriction as the overlays
    eligible = set(scored_paths(model_name, results_dir, repetitions))
    excluded = sorted({extract_prompt_cot(p)[0] for p in paths if p not in eligible})
    if excluded:
        print(f"Excluding prompts outside the current eligible set: {excluded}")
    return [p for p in paths if p in eligible]


def main():
    args = parse_args()
    paths = eligible_paths(args.model_name, args.results_dir, args.repetitions, args.all_files)
    if not paths:
        print(f"No quadrant files for {args.model_name}")
    else:
        print(f"Found {len(paths)} file(s)")
        plot_matrix(paths, args)


if __name__ == "__main__":
    main()

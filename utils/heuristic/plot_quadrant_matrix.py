'''
Plot heatmap of mean StrongReject score over prompt × normalised CoT sentence position,
plus line graphs of aggregated mean and standard deviation across prompts.

Each row is one (prompt, cot) pair from a CSV file.
Rows are clustered by prompt index, with horizontal borders separating groups
and square brackets on the y-axis labelling each cluster with its prompt index.

uv run -m utils.heuristic.plot_quadrant_matrix --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --no_colorbar
uv run -m utils.heuristic.plot_quadrant_matrix --model_name openai/gpt-oss-20b
'''

import argparse
import glob
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
    parser.add_argument("--random_sample", action="store_true",
                        help="Trim heatmap to ~3 random examples per model")
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


def plot_matrix(input_paths: list[str], args):
    if not input_paths:
        print("No data to plot.")
        return

    # ── Parse prompt/cot indices and group by prompt ──
    file_info = []
    for path in input_paths:
        prompt_idx, cot_idx = extract_prompt_cot(path)
        file_info.append({'path': path, 'prompt_idx': prompt_idx, 'cot_idx': cot_idx})

    file_info.sort(key=lambda x: (x['prompt_idx'], x['cot_idx']))


    all_prompts = list(dict.fromkeys(f['prompt_idx'] for f in file_info))

    if args.random_sample:
        h = 0.3

        # Line plot: N=32 random prompts
        sampled_line = set(np.random.choice(all_prompts, size=min(32, len(all_prompts)), replace=False))
        file_info_line = [f for f in file_info if f['prompt_idx'] in sampled_line]

        # Heatmap: 3 random prompts each with exactly 5 CoTs
        cot_counts = Counter(f['prompt_idx'] for f in file_info)
        prompts_with_5_cots = [p for p, c in cot_counts.items() if c == 5]
        sampled_heat = set(np.random.choice(
            prompts_with_5_cots, size=min(3, len(prompts_with_5_cots)), replace=False
        ))
        file_info_heat = [f for f in file_info if f['prompt_idx'] in sampled_heat]
    else:
        h = 0.05
        sampled_set = set(np.random.choice(all_prompts, size=min(32, len(all_prompts)), replace=False))
        file_info_heat = [f for f in file_info if f['prompt_idx'] in sampled_set]
        file_info_line = file_info_heat

    sorted_paths_heat = [f['path'] for f in file_info_heat]
    sorted_paths_line = [f['path'] for f in file_info_line]

    mean_mat, _ = build_matrices(sorted_paths_heat, args.n_bins)
    _, std_mat = build_matrices(sorted_paths_line, args.n_bins)

    # Cluster boundaries (based on heatmap data)
    prompt_indices_ordered = [f['prompt_idx'] for f in file_info_heat]
    unique_prompts = []
    cluster_boundaries = []
    for i, pi in enumerate(prompt_indices_ordered):
        if i == 0 or pi != prompt_indices_ordered[i - 1]:
            unique_prompts.append(pi)
            cluster_boundaries.append(i)
    cluster_boundaries.append(len(file_info_heat))

    n_rows = mean_mat.shape[0]
    bin_centres = np.linspace(0.0, 1.0, args.n_bins)

    # ── Figure layout ──
    heatmap_height = max(3, 0.35 * n_rows)
    line_height = 3.0
    panel_width = 7
    cbar_width = 0.4
    fig_width = panel_width + (0 if args.no_colorbar else cbar_width + 1.5)
    fig_height = heatmap_height + line_height
    fig = plt.figure(figsize=(fig_width, fig_height))
    if args.model_name == 'openai/gpt-oss-20b':
        model_short = 'GPT-OSS-20B'
    else:
        model_short = args.model_name.split('/')[-1]
    title_margin_inches = 0
    top = 1.0 - title_margin_inches / fig_height
    fig.suptitle(model_short, fontsize=22, fontweight='bold', y=top + 0.07)
    fig.subplots_adjust(top=top)



    if args.no_colorbar:
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[line_height, heatmap_height],
            hspace=h
        )
    else:
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[line_height, heatmap_height],
            width_ratios=[1, cbar_width / panel_width],
            hspace=h, wspace=0.2
        )

    # ── Row 1: Heatmap ──
    ax_heat = fig.add_subplot(gs[1, 0])
    im = ax_heat.imshow(mean_mat, aspect='auto',
                        cmap='Reds', vmin=0.0, vmax=1.0,
                        interpolation='nearest')

    # X-axis
    tick_positions_x = np.linspace(0, args.n_bins - 1, 6)
    tick_labels_x = [f'{v:.1f}' for v in np.linspace(0, 1, 6)]
    ax_heat.set_xticks(tick_positions_x)
    ax_heat.set_xticklabels(tick_labels_x, fontsize=22)
    ax_heat.set_xlabel('Normalised CoT Sentence Position', fontsize=22)
    ax_heat.tick_params(axis='x', labelsize=22)

    # Y-axis: square brackets grouping rows by prompt
    ax_heat.set_yticks([])
    if args.no_colorbar:
        ax_heat.set_ylabel('Prompt Index', fontsize=22, labelpad=65)

    trans = mtransforms.blended_transform_factory(ax_heat.transAxes, ax_heat.transData)
    x_tip = -0.012
    x_base = -0.028
    x_text = -0.055

    gap = 0.15  # half-gap in data (row) units

    for g, pi in enumerate(unique_prompts):
        start = cluster_boundaries[g] - 0.5 + gap
        end = cluster_boundaries[g + 1] - 0.5 - gap
        centre = (start + end) / 2.0

        ax_heat.plot(
            [x_base, x_tip, x_tip, x_base],
            [start, start, end, end],
            transform=trans, color='black', linewidth=1.5,
            clip_on=False, solid_capstyle='butt',
        )
        ax_heat.text(
            x_text, centre, str(pi),
            transform=trans, ha='right', va='center', fontsize=18,
        )

    # White separators between prompt clusters
    for boundary in cluster_boundaries[1:-1]:
        ax_heat.axhline(y=boundary - 0.5, color='white', linewidth=5.0)

    # Black border around each prompt cluster (no continuous outer spine)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    for g in range(len(unique_prompts)):
        y0 = cluster_boundaries[g] - 0.5 + (gap if g > 0 else 0)
        y1 = cluster_boundaries[g + 1] - 0.5 - (gap if g < len(unique_prompts) - 1 else 0)
        rect = mpatches.Rectangle(
            (-0.5, y0), args.n_bins, y1 - y0,
            linewidth=1.5, edgecolor='black', facecolor='none'
        )
        ax_heat.add_patch(rect)

    # Colorbar
    if not args.no_colorbar:
        cbar_ax = fig.add_subplot(gs[1, 1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Mean StrongREJECT Score', fontsize=22)
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.annotate('comply', xy=(0.5, 1.08), xycoords='axes fraction',
                         fontsize=18, fontstyle='italic', ha='center')
        cbar.ax.annotate('refuse', xy=(0.5, -0.08), xycoords='axes fraction',
                         fontsize=18, fontstyle='italic', ha='center')

    # ── Row 0: Std across rollouts averaged across files ──
    ax_line = fig.add_subplot(gs[0, 0])

    avg_std = np.nanmean(std_mat, axis=0)
    std_of_std = np.nanstd(std_mat, axis=0)

    ax_line.plot(bin_centres, avg_std, linewidth=2, color='#2166ac')
    ax_line.fill_between(bin_centres,
                         avg_std - std_of_std,
                         avg_std + std_of_std,
                         alpha=0.25, color='#2166ac')
    ax_line.set_xlim(0, 1)
    ax_line.set_ylim(0, 0.5)
    ax_line.set_xlabel('Normalised CoT Sentence Position', fontsize=22)
    if args.no_colorbar:
        ax_line.set_ylabel('Mean Std Dev\n(across rollouts)', fontsize=20)
    ax_line.tick_params(labelsize=22)
    ax_line.grid(True, alpha=0.3)

    # Save
    out_dir = os.path.join(args.results_dir, args.model_name, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    if args.random_sample:
        output_path = os.path.join(
            out_dir,
            f'heatmap_quadrant_rep{args.repetitions}_bins{args.n_bins}_sampled.pdf'
        )
    else:
        output_path = os.path.join(
            out_dir,
            f'heatmap_quadrant_rep{args.repetitions}_bins{args.n_bins}.pdf'
        )
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def main():
    args = parse_args()

    input_csv = f'scored_full_resample_prompt*_cot*_rep_{args.repetitions}.csv'

    pattern_quadrant = os.path.join(
        args.results_dir, args.model_name, 'dataset', 'quadrants', input_csv
    )
    input_paths_quadrant = sorted(glob.glob(pattern_quadrant))

    if not input_paths_quadrant:
        print(f"No files found matching: {pattern_quadrant}")
    else:
        print(f"Found {len(input_paths_quadrant)} file(s)")
        plot_matrix(input_paths_quadrant, args)


if __name__ == "__main__":
    main()
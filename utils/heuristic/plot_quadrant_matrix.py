'''
Plot heatmap of mean StrongReject score over prompt × normalised CoT sentence position,
plus line graphs of aggregated mean and standard deviation across prompts.

Each row is one (prompt, cot) pair from a CSV file.
Rows are clustered by prompt index, with horizontal borders separating groups.
Sentence indices are normalised to [0, 1] and interpolated so that traces
of different lengths are comparable.

Example usage:
uv run -m utils.heuristic.plot_quadrant_matrix \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --repetitions 10
'''

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
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

    For each CSV (one prompt+cot's resampling rollouts):
      1. Group by sentence_idx, compute mean and std across repetitions.
      2. Normalise sentence_idx to [0, 1].
      3. Linearly interpolate both onto a uniform grid of n_bins points.

    Returns:
        mean_matrix:  (num_files, n_bins) — mean score per sentence per file.
        std_matrix:   (num_files, n_bins) — std of scores per sentence per file.
    """
    bin_centres = np.linspace(0.0, 1.0, n_bins)

    mean_matrix = np.full((len(input_paths), n_bins), np.nan)
    std_matrix = np.full((len(input_paths), n_bins), np.nan)

    for row_idx, input_path in enumerate(input_paths):
        df = pd.read_csv(input_path)

        max_idx = df['sentence_idx'].max()
        if max_idx == 0:
            # Only one sentence position — fill with constant
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
    """
    Plot a 2-row figure:
      Row 0: heatmap of per-(prompt, cot) mean score, clustered by prompt index.
      Row 1: line graph of per-file std averaged across all files (± std across files).

    Rows within each prompt cluster are sorted by cot index.
    Horizontal borders separate prompt clusters.
    Y-axis labels show the prompt index.
    """
    if not input_paths:
        print("No data to plot.")
        return

    # ── Parse prompt/cot indices and group by prompt ──
    file_info = []
    for path in input_paths:
        prompt_idx, cot_idx = extract_prompt_cot(path)
        file_info.append({'path': path, 'prompt_idx': prompt_idx, 'cot_idx': cot_idx})

    # Sort: primary by prompt_idx, secondary by cot_idx
    file_info.sort(key=lambda x: (x['prompt_idx'], x['cot_idx']))
    sorted_paths = [f['path'] for f in file_info]

    # Build matrices in the clustered order
    mean_mat, std_mat = build_matrices(sorted_paths, args.n_bins)

    # Compute cluster boundaries (for borders) and tick positions (for labels)
    prompt_indices_ordered = [f['prompt_idx'] for f in file_info]
    unique_prompts = []
    cluster_boundaries = []  # row indices where a new prompt group starts
    for i, pi in enumerate(prompt_indices_ordered):
        if i == 0 or pi != prompt_indices_ordered[i - 1]:
            unique_prompts.append(pi)
            cluster_boundaries.append(i)
    cluster_boundaries.append(len(file_info))  # sentinel for the last group

    # Tick position = centre of each cluster; label = prompt index
    ytick_positions = []
    ytick_labels = []
    for g, pi in enumerate(unique_prompts):
        start = cluster_boundaries[g]
        end = cluster_boundaries[g + 1]
        ytick_positions.append((start + end - 1) / 2.0)
        ytick_labels.append(str(pi))

    n_rows = mean_mat.shape[0]
    bin_centres = np.linspace(0.0, 1.0, args.n_bins)

    # ── Figure layout: 2 rows × 2 cols (heatmap + colorbar, line plot + empty) ──
    heatmap_height = max(3, 0.35 * n_rows)
    line_height = 3.0
    panel_width = 7
    cbar_width = 0.4
    fig_width = panel_width + cbar_width + 1.5
    fig_height = heatmap_height + line_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[heatmap_height, line_height],
        width_ratios=[1, cbar_width / panel_width],
        hspace=0.15, wspace=0.08
    )

    # ── Row 0: Heatmap ──
    ax_heat = fig.add_subplot(gs[0, 0])
    im = ax_heat.imshow(mean_mat, aspect='auto',
                        cmap='RdYlGn_r', vmin=0.0, vmax=1.0,
                        interpolation='nearest')

    # X-axis: normalised position
    tick_positions_x = np.linspace(0, args.n_bins - 1, 6)
    tick_labels_x = [f'{v:.1f}' for v in np.linspace(0, 1, 6)]
    ax_heat.set_xticks(tick_positions_x)
    ax_heat.set_xticklabels(tick_labels_x)
    ax_heat.set_xlabel('Normalised CoT Sentence Position', fontsize=11)

    # Y-axis: prompt index labels at cluster centres
    ax_heat.set_yticks(ytick_positions)
    ax_heat.set_yticklabels(ytick_labels, fontsize=9)
    ax_heat.set_ylabel('Prompt Index', fontsize=11)

    # Draw horizontal borders between prompt clusters
    for boundary in cluster_boundaries[1:-1]:  # skip first (top) and sentinel (bottom)
        ax_heat.axhline(y=boundary - 0.5, color='white', linewidth=2.5)

    # ax_heat.set_title('Mean StrongReject Score by CoT Position', fontsize=12)

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Mean StrongReject Score', fontsize=11)
    cbar.ax.annotate('comply', xy=(0.5, 1.02), xycoords='axes fraction',
                     fontsize=9, fontstyle='italic', ha='center')
    cbar.ax.annotate('refuse', xy=(0.5, -0.03), xycoords='axes fraction',
                     fontsize=9, fontstyle='italic', ha='center')

    # ── Row 1: Std (across repetitions) averaged across all files ──
    ax_line = fig.add_subplot(gs[1, 0])

    avg_std = np.nanmean(std_mat, axis=0)
    std_of_std = np.nanstd(std_mat, axis=0)

    ax_line.plot(bin_centres, avg_std, linewidth=2, color='#2166ac')
    ax_line.fill_between(bin_centres,
                         avg_std - std_of_std,
                         avg_std + std_of_std,
                         alpha=0.25, color='#2166ac')
    ax_line.set_xlim(0, 1)
    ax_line.set_ylim(0, 0.5)
    ax_line.set_xlabel('Normalised CoT Sentence Position', fontsize=11)
    ax_line.set_ylabel('Mean Std Dev\n(across rollouts)', fontsize=10)
    ax_line.grid(True, alpha=0.3)

    # Save
    out_dir = os.path.join(args.results_dir, args.model_name, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir,
        f'heatmap_quadrant_rep{args.repetitions}_bins{args.n_bins}.png'
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
'''
Plot heatmap of mean StrongReject score over prompt × normalised CoT sentence position,
plus line graphs of aggregated mean and standard deviation across prompts.

Each row is a prompt (one CSV file from the resampling pipeline).
Sentence indices are normalised to [0, 1] and interpolated so that traces
of different lengths are comparable.

Example usage:
uv run -m utils.heuristic.plot_matrix \
    --model_name openai/gpt-oss-20b \
    --repetitions 10 \
    --n_bins 30
'''

import argparse
import glob
import os

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


def build_matrices(input_paths: list[str],
                   n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (num_prompts × n_bins) matrices of per-sentence mean and std scores.

    For each CSV (one prompt's resampling rollouts):
      1. Group by sentence_idx, compute mean and std across repetitions.
      2. Normalise sentence_idx to [0, 1].
      3. Linearly interpolate both onto a uniform grid of n_bins points.

    Returns:
        mean_matrix:  (num_prompts, n_bins) — mean score per sentence per prompt.
        std_matrix:   (num_prompts, n_bins) — std of scores per sentence per prompt.
    """
    bin_centres = np.linspace(0.0, 1.0, n_bins)

    mean_matrix = np.full((len(input_paths), n_bins), np.nan)
    std_matrix = np.full((len(input_paths), n_bins), np.nan)

    for row_idx, input_path in enumerate(input_paths):
        df = pd.read_csv(input_path)

        max_idx = df['sentence_idx'].max()
        df['norm_pos'] = df['sentence_idx'] / max_idx

        mean_at_pos = df.groupby('norm_pos')['strongreject_score'].mean().sort_index()
        mean_at_pos = mean_at_pos.dropna()  # drop positions with no valid scores
        std_at_pos = df.groupby('norm_pos')['strongreject_score'].std().fillna(0).sort_index()
        std_at_pos = std_at_pos.loc[mean_at_pos.index]  # keep only matching positions

        mean_matrix[row_idx, :] = np.interp(
            bin_centres, mean_at_pos.index.values, mean_at_pos.values
        )
        std_matrix[row_idx, :] = np.interp(
            bin_centres, std_at_pos.index.values, std_at_pos.values
        )

        # if np.isnan(mean_matrix[row_idx]).any():
        #     print(f"WARNING: NaN bins in {input_path}")
        #     nan_in_means = mean_at_pos[mean_at_pos.isna()]
        #     if not nan_in_means.empty:
        #         print(f"  NaN group means at norm_pos: {nan_in_means.index.tolist()}")

    return mean_matrix, std_matrix


def plot_matrix(input_paths_refusal: list[str],
                input_paths_nonrefusal: list[str],
                args):
    """
    Plot a 2-row × 2-col figure:
      Row 0: heatmaps of per-prompt mean score
      Row 1: line graph of per-prompt std averaged across prompts (± std across prompts)
    """
    panel_data = []
    for paths, category in [(input_paths_refusal, 'Refusal'),
                            (input_paths_nonrefusal, 'Non-refusal')]:
        if not paths:
            continue
        mean_mat, std_mat = build_matrices(paths, args.n_bins)
        # Sort rows by final-bin score for the heatmap
        final_scores = np.nanmean(mean_mat[:, -3:], axis=1)
        sort_order = np.argsort(final_scores)
        mean_mat_sorted = mean_mat[sort_order]
        panel_data.append({
            'category': category,
            'mean_matrix': mean_mat,
            'mean_matrix_sorted': mean_mat_sorted,
            'std_matrix': std_mat,
        })

    if not panel_data:
        print("No data to plot.")
        return

    n_panels = len(panel_data)
    max_rows = max(p['mean_matrix'].shape[0] for p in panel_data)
    bin_centres = np.linspace(0.0, 1.0, args.n_bins)

    # Layout: 2 rows × (n_panels + 1) cols — extra narrow column for colorbar
    # This ensures heatmaps and line plots share identical panel widths.
    heatmap_height = max(3, 0.2 * max_rows)
    line_height = 3.0
    panel_width = 6
    cbar_width = 0.4
    fig_width = panel_width * n_panels + cbar_width + 2
    fig_height = heatmap_height + line_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2, n_panels + 1,
        height_ratios=[heatmap_height, line_height],
        width_ratios=[1] * n_panels + [cbar_width / panel_width],
        hspace=0.15, wspace=0.3
    )

    # ── Row 0: Heatmaps ──
    heatmap_axes = []
    for col, panel in enumerate(panel_data):
        ax = fig.add_subplot(gs[0, col])
        heatmap_axes.append(ax)
        im = ax.imshow(panel['mean_matrix_sorted'], aspect='auto',
                       cmap='RdYlGn_r', vmin=0.0, vmax=1.0,
                       interpolation='nearest')

        tick_positions = np.linspace(0, args.n_bins - 1, 6)
        tick_labels_x = [f'{v:.1f}' for v in np.linspace(0, 1, 6)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_x)
        ax.set_xlabel('Normalised CoT Sentence Position', fontsize=11)

        n_rows = panel['mean_matrix_sorted'].shape[0]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(range(1, n_rows + 1), fontsize=8)
        if col == 0:
            ax.set_ylabel('Prompt', fontsize=11)

        ax.set_title(f'{panel["category"]} prompts', fontsize=12)

    # Share y-axis across heatmap columns
    if n_panels > 1:
        heatmap_axes[1].sharey(heatmap_axes[0])

    # Colorbar in its dedicated column
    cbar_ax = fig.add_subplot(gs[0, n_panels])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Mean StrongReject Score', fontsize=11)
    cbar.ax.annotate('comply', xy=(0.5, 1.02), xycoords='axes fraction',
                     fontsize=9, fontstyle='italic', ha='center')
    cbar.ax.annotate('refuse', xy=(0.5, -0.03), xycoords='axes fraction',
                     fontsize=9, fontstyle='italic', ha='center')

    # ── Row 1: Std (across repetitions) averaged across prompts ──
    for col, panel in enumerate(panel_data):
        ax = fig.add_subplot(gs[1, col])
        std_mat = panel['std_matrix']

        avg_std = np.nanmean(std_mat, axis=0)
        std_of_std = np.nanstd(std_mat, axis=0)

        ax.plot(bin_centres, avg_std, linewidth=2, color='#2166ac')
        ax.fill_between(bin_centres,
                        avg_std - std_of_std,
                        avg_std + std_of_std,
                        alpha=0.25, color='#2166ac')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel('Normalised CoT Sentence Position', fontsize=11)
        if col == 0:
            ax.set_ylabel('Mean Std Dev\n(across rollouts)', fontsize=10)
        ax.grid(True, alpha=0.3)

    # # ── Suptitle ──
    # model_short = args.model_name.split('/')[-1]
    # fig.suptitle(
    #     f'{model_short}\n'
    #     f'Mean output StrongReject score by normalised CoT position  '
    #     f'(k={args.repetitions} rollouts per sentence)',
    #     fontsize=15, y=1.0
    # )

    # Save
    out_dir = os.path.join(args.results_dir, args.model_name, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir,
        f'heatmap_combined_rep{args.repetitions}_bins{args.n_bins}.png'
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to: {output_path}")


def main():
    args = parse_args()

    input_csv = f'scored_full_resample_*_rep_{args.repetitions}.csv'

    pattern_refusal = os.path.join(
        args.results_dir, args.model_name, 'dataset', 'refusal', input_csv
    )
    pattern_nonrefusal = os.path.join(
        args.results_dir, args.model_name, 'dataset', 'nonrefusal', input_csv
    )

    input_paths_refusal = sorted(glob.glob(pattern_refusal))
    input_paths_nonrefusal = sorted(glob.glob(pattern_nonrefusal))

    if not input_paths_refusal:
        print(f"No files found matching: {pattern_refusal}")
    else:
        print(f"Found {len(input_paths_refusal)} refusal file(s)")

    if not input_paths_nonrefusal:
        print(f"No files found matching: {pattern_nonrefusal}")
    else:
        print(f"Found {len(input_paths_nonrefusal)} non-refusal file(s)")

    plot_matrix(input_paths_refusal, input_paths_nonrefusal, args)


if __name__ == "__main__":
    main()
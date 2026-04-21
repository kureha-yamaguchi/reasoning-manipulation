'''
Plot standard deviation of StrongReject score vs normalised CoT depth,
with one line per prompt (averaged over CoT prefixes) and a bold mean line.

uv run -m utils.heuristic.plot_lines --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
uv run -m utils.heuristic.plot_lines --model_name openai/gpt-oss-20b
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
        description="Plot std dev of StrongReject score vs normalised CoT depth per prompt"
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


def build_std_matrix(input_paths: list[str], n_bins: int) -> np.ndarray:
    """
    Build (num_files × n_bins) matrix of per-normalised-position std scores.
    """
    bin_centres = np.linspace(0.0, 1.0, n_bins)
    std_matrix = np.full((len(input_paths), n_bins), np.nan)

    for row_idx, input_path in enumerate(input_paths):
        df = pd.read_csv(input_path)

        max_idx = df['sentence_idx'].max()
        if max_idx == 0:
            std_val = df['strongreject_score'].std()
            std_matrix[row_idx, :] = std_val if not np.isnan(std_val) else 0.0
            continue

        df['norm_pos'] = df['sentence_idx'] / max_idx
        std_at_pos = df.groupby('norm_pos')['strongreject_score'].std().fillna(0).sort_index()

        std_matrix[row_idx, :] = np.interp(
            bin_centres, std_at_pos.index.values, std_at_pos.values
        )

    return std_matrix


def plot_per_prompt_std(input_paths: list[str], args):
    if not input_paths:
        print("No data to plot.")
        return

    file_info = []
    for path in input_paths:
        prompt_idx, cot_idx = extract_prompt_cot(path)
        file_info.append({'path': path, 'prompt_idx': prompt_idx, 'cot_idx': cot_idx})

    file_info.sort(key=lambda x: (x['prompt_idx'], x['cot_idx']))

    sorted_paths = [f['path'] for f in file_info]
    std_mat = build_std_matrix(sorted_paths, args.n_bins)

    # Group rows by prompt index → mean std per prompt
    prompt_indices = [f['prompt_idx'] for f in file_info]
    unique_prompts = list(dict.fromkeys(prompt_indices))

    prompt_std = np.full((len(unique_prompts), args.n_bins), np.nan)
    for i, pi in enumerate(unique_prompts):
        rows = [r for r, p in enumerate(prompt_indices) if p == pi]
        prompt_std[i, :] = np.nanmean(std_mat[rows, :], axis=0)

    bin_centres = np.linspace(0.0, 1.0, args.n_bins)

    if args.model_name == 'openai/gpt-oss-20b':
        model_short = 'GPT-OSS-20B'
    else:
        model_short = args.model_name.split('/')[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(model_short, fontsize=18, fontweight='bold')

    cmap = plt.get_cmap('tab20', len(unique_prompts))
    for i, pi in enumerate(unique_prompts):
        ax.plot(bin_centres, prompt_std[i], linewidth=1.0,
                color=cmap(i), alpha=0.6, label=f'prompt {pi}')

    # Bold mean line across all prompts
    mean_std = np.nanmean(prompt_std, axis=0)
    ax.plot(bin_centres, mean_std, linewidth=2.5, color='black',
            label='mean', zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel('Normalised CoT Depth', fontsize=14)
    ax.set_ylabel('Std Dev of StrongREJECT Score', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc='upper right', fontsize=7, ncol=2,
        framealpha=0.7, title='Prompt', title_fontsize=8
    )

    plt.tight_layout()

    out_dir = os.path.join(args.results_dir, args.model_name, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir,
        f'std_per_prompt_rep{args.repetitions}_bins{args.n_bins}.pdf'
    )
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    args = parse_args()

    input_csv = f'scored_full_resample_prompt*_cot*_rep_{args.repetitions}.csv'
    pattern = os.path.join(
        args.results_dir, args.model_name, 'dataset', 'quadrants', input_csv
    )
    input_paths = sorted(glob.glob(pattern))

    if not input_paths:
        print(f"No files found matching: {pattern}")
    else:
        print(f"Found {len(input_paths)} file(s)")
        plot_per_prompt_std(input_paths, args)


if __name__ == "__main__":
    main()

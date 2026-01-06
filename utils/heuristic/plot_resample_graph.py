'''
Plot average strongreject score as a function of sentence_idx.

Example usage:
uv run -m utils.heuristic.plot_resampling_scores \
    --input resampling_results_idx3_cot1_s1_e30.csv \
    --output score_plot.png
'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average strongreject score vs sentence index"
    )
    parser.add_argument("--model_name", type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                        help="Path to scored CSV file")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to scored CSV file")
    parser.add_argument("--output", type=str, default="score_plot.png",
                        help="Output path for the plot")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.join('results', args.model_name, 'dataset', args.input)
    base_name = os.path.splitext(args.input)[0]
    output_path = os.path.join('results', args.model_name, 'figures', f"graph_{base_name}.csv")
    # Load data
    df = pd.read_csv(input_path)
    
    # Compute mean strongreject_score per sentence_idx
    mean_scores = df.groupby('sentence_idx')['strongreject_score'].mean()
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_scores.index, mean_scores.values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Sentence Index', fontsize=12)
    plt.ylabel('Mean StrongReject Score', fontsize=12)
    plt.title('Average StrongReject Score by Sentence Index', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
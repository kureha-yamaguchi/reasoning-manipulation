'''
Plot average strongreject score as a function of sentence_idx.

Example usage:
uv run -m utils.heuristic.plot_resample_graph \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --index_number 3 \
    --cot_number 1
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
    parser.add_argument("--index_number", type=int, required=True,
                        help="Index number as per quadrant_output.txt")
    parser.add_argument("--cot_number", type=int, required=True,
                        help="Cot number as per quadrant_output.txt (x/5)")
    parser.add_argument("--repetitions", type=int, default=15,
                        help="Number of output variations per prompt")
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = f'scored_resampling_results_idx{args.index_number}_cot{args.cot_number}.csv'
    input_path = os.path.join('results', args.model_name, 'dataset', input_csv)
    base_name = os.path.splitext(input_csv)[0]
    output_path = os.path.join('results', args.model_name, 'figures', f'graph_{base_name}')
    # Load data
    df = pd.read_csv(input_path)
    
    # Compute mean and std strongreject_score per sentence_idx
    prompt = df['prompt'][0]
    if len(prompt) > 60:
        prompt = prompt[:60] + '...'
    mean_scores = df.groupby('sentence_idx')['strongreject_score'].mean()
    std_scores = df.groupby('sentence_idx')['strongreject_score'].std().fillna(0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_scores.index, mean_scores.values, marker='o', linewidth=2, markersize=6, label='Mean')
    plt.fill_between(mean_scores.index, 
                     mean_scores.values - std_scores.values, 
                     mean_scores.values + std_scores.values, 
                     alpha=0.3, label='±1 Std Dev')
    plt.xlabel('CoT Sentence Index', fontsize=12)
    plt.ylabel('Mean Output StrongReject Score After Resampling', fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.title(f'StrongReject Score & Similarity by Sentence Index\n'
            f'Prompt: {prompt}\n'
            f'CoT rollout: {args.cot_number}\nNumber of resampling repetitions: {args.repetitions}', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
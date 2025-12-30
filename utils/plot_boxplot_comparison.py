"""
Generate box plots comparing StrongReject scores before and after intervention.

This module creates box plots to visualize the distribution of strongreject_score
values from baseline test data versus intervention results. It compares scores
from the original test dataset against scores from ortho model outputs.

Example usage:
    uv run -m utils.plot_boxplot_comparison \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --type cot \
        --layer 17
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate box plot comparing strongreject_score before and after intervention"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
    parser.add_argument("--type", type=str, required=True,
                        help="Type of intervention (e.g., cot, baseline, all)")
    parser.add_argument("--layer", type=str, required=True,
                        help="Layer number (e.g., 16)")
    parser.add_argument("--harmless", action='store_true',
                        help='For harmless nonrefusal, harmful refusal dataset configuration')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Construct file paths
    before_path = f"results/{args.model_name}/dataset/scored_test_harmful_prompts_cot5_out5.csv"

    if args.type == 'all':
        # Split the comma-separated layer numbers
        layers = args.layer.split(',')
        cot_layer = layers[0].strip()
        baseline_layer = layers[1].strip()

        if args.harmless:
            cot_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}_harmless.csv"
            baseline_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layer}_harmless.csv"
        else:
            cot_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}.csv"
            baseline_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layer}.csv"
        
        # Load data
        before_df = pd.read_csv(before_path)
        cot_df = pd.read_csv(cot_path)
        baseline_df = pd.read_csv(baseline_path)
        
        # Extract strongreject_score columns
        before_scores = before_df['strongreject_score'].dropna()
        cot_scores = cot_df['strongreject_score'].dropna()
        baseline_scores = baseline_df['strongreject_score'].dropna()
        
        # Create comparison dataframe
        comparison_data = pd.DataFrame({
            'Score': list(before_scores) + list(cot_scores) + list(baseline_scores),
            'Group': ['Before'] * len(before_scores) + ['After (cot)'] * len(cot_scores) + ['After (baseline)'] * len(baseline_scores)
        })
        
        # Create box plot (hide outliers)
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='Score', y='Group', data=comparison_data, width=0.5,
                        order=['After (cot)', 'After (baseline)', 'Before'],
                        palette={'After (baseline)': '#ff9900', 'After (cot)': '#ff9900', 'Before': '#50a9a9'},
                        showfliers=False)
        # Adjust spacing for 3 groups (y positions: 0, 1, 2)
        ax.set_ylim(-0.5, 2.5)
        plt.title(f'StrongReject Score Comparison: {args.type.upper()}, Layer (baseline: {baseline_layer}, cot: {cot_layer})\n Model:{args.model_name} \n Rollouts (5 cot 5 output) per prompt in holdout set (487 harmful prompts)', fontsize=14)
        plt.xlabel('StrongReject Score', fontsize=12)
        plt.ylabel('')
        plt.tight_layout()

        # Save plot
        output_dir = f"results/{args.model_name}/figures"
        os.makedirs(output_dir, exist_ok=True)

        if args.harmless:
            output_file = f"{output_dir}/boxplot_comparison_{args.type}_layer_{cot_layer}_{baseline_layer}_harmless.png"
        else:
            output_file = f"{output_dir}/boxplot_comparison_{args.type}_layer_{cot_layer}_{baseline_layer}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to: {output_file}")
    else:
        after_path = f"results/{args.model_name}/attack_results/scored_ortho_output_test_harmful_prompts_{args.type}_layer_{args.layer}.csv"
        
        # Load data
        before_df = pd.read_csv(before_path)
        after_df = pd.read_csv(after_path)
        
        # Extract strongreject_score columns
        before_scores = before_df['strongreject_score'].dropna()
        after_scores = after_df['strongreject_score'].dropna()
        
        # Create comparison dataframe
        comparison_data = pd.DataFrame({
            'Score': list(before_scores) + list(after_scores),
            'Group': ['Before'] * len(before_scores) + ['After'] * len(after_scores)
        })
        
        # Create box plot (hide outliers)
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='Score', y='Group', data=comparison_data, width=0.5,
                        order=['After', 'Before'],
                        palette={'Before': '#50a9a9', 'After': '#ff9900'},
                        showfliers=False)
        # Reduce spacing between boxes
        ax.set_ylim(-0.5, 1.5)
        plt.title(f'StrongReject Score Comparison: {args.type.upper()}, Layer {args.layer}\n Model:{args.model_name} \n Rollouts (5 cot 5 output) per prompt in holdout set (487 harmful prompts)', fontsize=14)
        plt.xlabel('StrongReject Score', fontsize=12)
        plt.ylabel('')
        plt.tight_layout()
    
        # Save plot
        output_dir = f"results/{args.model_name}/figures"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/boxplot_comparison_{args.type}_layer_{args.layer}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to: {output_file}")


if __name__ == "__main__":
    main()


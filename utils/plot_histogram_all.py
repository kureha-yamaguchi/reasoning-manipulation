"""
Generate histogram comparing StrongReject scores before and after intervention across all models.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    
    MODEL1 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    MODEL2 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MODEL3 = "Qwen/Qwen3-8B"

    model_names = [MODEL1, MODEL2, MODEL3]
    # Shorter display names for the plot
    display_names = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-7B", "Qwen3-8B"]
    
    cot_layers = [17,17,17]
    baseline_layers = [11,17,17]
    
    all_data = []
    
    for model_name, display_name, cot_layer, baseline_layer in zip(model_names, display_names, cot_layers, baseline_layers):
        # Construct file paths
        before_path = f"results/{model_name}/dataset/scored_test_harmful_prompts_cot5_out5.csv"
        cot_path = f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_cot_layer_{cot_layer}.csv"
        baseline_path = f"results/{model_name}/attack_results/scored_ortho_output_test_harmful_prompts_baseline_layer_{baseline_layer}.csv"
        
        # Load data
        before_df = pd.read_csv(before_path)
        cot_df = pd.read_csv(cot_path)
        baseline_df = pd.read_csv(baseline_path)
        
        # Extract scores
        before_scores = before_df['strongreject_score'].dropna()
        cot_scores = cot_df['strongreject_score'].dropna()
        baseline_scores = baseline_df['strongreject_score'].dropna()
        
        # Add to combined dataframe with model info

        display = f"{display_name} \n Layers (baseline: {baseline_layer}, cot: {cot_layer})"
        for score in before_scores:
            all_data.append({'Score': score, 'Condition': 'Before intervention', 'Model': display})
        for score in baseline_scores:
            all_data.append({'Score': score, 'Condition': 'After (baseline)', 'Model': display})
        for score in cot_scores:
            all_data.append({'Score': score, 'Condition': 'After (cot)', 'Model': display})
    
    combined_df = pd.DataFrame(all_data)
    
    # Create the plot with hue for conditions
    plt.figure(figsize=(14, 6))
    ax = sns.boxplot(
        data=combined_df,
        x='Model',
        y='Score',
        hue='Condition',
        hue_order=['Before intervention', 'After (baseline)', 'After (cot)'],
        palette={'Before intervention': '#50a9a9', 'After (baseline)': '#cc6600', 'After (cot)': '#ff9900', },
        showfliers=False
    )
    
    plt.title(f'StrongReject score comparison across models on held-out test set, before and after weight orthogonalisation', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('StrongReject Score', fontsize=12)
    plt.legend(title='Condition', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    # Save plot
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/boxplot_all_models.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {output_file}")


if __name__ == "__main__":
    main()
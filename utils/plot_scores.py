#!/usr/bin/env python3
"""
Script to generate layered spectrograms of strongreject_score values from CSV files.
Reads files with format: prompt,cot,output,cot_rep_n,output_rep_n,strongreject_score
Creates smooth density plots with different colors for each model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def extract_model_name(file_path):
    """Extract model name from file path structure."""
    path = Path(file_path)
    # Navigate up to find the model directory (should be 2 levels up from the CSV file)
    # Structure: results/Vendor/Model/dataset/file.csv
    parts = path.parts
    try:
        # Find 'dataset' in path and get the parent directory name
        dataset_idx = parts.index('dataset')
        model_name = parts[dataset_idx - 1]  # Model name is parent of 'dataset'
        return model_name
    except (ValueError, IndexError):
        # Fallback: try to extract from parent directories
        if len(parts) >= 3:
            return parts[-3]  # Third from last part
        return path.stem  # Use filename without extension as fallback

def load_and_process_csv(file_path):
    """Load CSV file and extract strongreject_score values."""
    try:
        df = pd.read_csv(file_path)
        
        # Check if required column exists
        if 'strongreject_score' not in df.columns:
            print(f"Warning: 'strongreject_score' column not found in {file_path}")
            return None, None
        
        # Extract model name
        model_name = extract_model_name(file_path)
        
        # Get strongreject_score values, removing any NaN values
        scores = df['strongreject_score'].dropna().values
        
        print(f"Loaded {len(scores)} scores for model: {model_name}")
        print(f"Score range: {scores.min():.4f} to {scores.max():.4f}")
        
        return model_name, scores
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def create_spectrogram(data_dict, output_file='strongreject_spectrogram.png', 
                      figsize=(12, 8), alpha=0.7, bandwidth=0.02, log_scale=False, 
                      use_histogram=False, bins=50):
    """
    Create layered spectrogram (density plot) of strongreject scores.
    
    Args:
        data_dict: Dictionary with model_name -> scores mapping
        output_file: Output filename for the plot
        figsize: Figure size tuple
        alpha: Transparency of density lines
        bandwidth: KDE bandwidth for smoothing
        log_scale: Whether to use log scale for y-axis (density)
        use_histogram: Use histogram instead of KDE for discrete data
        bins: Number of bins for histogram (when use_histogram=True)
    """
    plt.figure(figsize=figsize)
    
    # Set up color palette
    colors = plt.cm.Set1(np.linspace(0, 1, len(data_dict)))
    
    # Create density plots for each model
    for i, (model_name, scores) in enumerate(data_dict.items()):
        if len(scores) == 0:
            continue
        
        print(f"Processing {model_name}: unique values = {len(np.unique(scores))}, total = {len(scores)}")
            
        if use_histogram:
            # Use histogram for discrete/categorical data
            counts, bin_edges, patches = plt.hist(scores, bins=bins, alpha=alpha, 
                                                color=colors[i], density=True, 
                                                label=f'{model_name} (n={len(scores)})',
                                                histtype='step', linewidth=2.5)
        else:
            # Create KDE for smooth density estimation
            try:
                # Check if data is too discrete for KDE
                unique_vals = len(np.unique(scores))
                if unique_vals < 10:
                    print(f"Warning: {model_name} has only {unique_vals} unique values. Consider using --histogram flag.")
                
                kde = stats.gaussian_kde(scores, bw_method=bandwidth)
                
                # Create x-axis points for smooth curve
                x_min, x_max = scores.min(), scores.max()
                x_range = x_max - x_min
                if x_range == 0:  # All values are identical
                    x_range = 0.1 if x_min == 0 else abs(x_min) * 0.1
                x_points = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
                
                # Calculate density
                density = kde(x_points)
                
                # Plot smooth density curve
                plt.plot(x_points, density, color=colors[i], 
                        label=f'{model_name} (n={len(scores)})', 
                        linewidth=2.5, alpha=alpha)
                
                # Fill under curve for better visualization
                plt.fill_between(x_points, density, alpha=0.2, color=colors[i])
                
            except Exception as e:
                print(f"Error creating KDE for {model_name}: {e}")
                # Fallback to histogram if KDE fails
                plt.hist(scores, bins=bins, alpha=0.3, color=colors[i], 
                        density=True, label=f'{model_name} (n={len(scores)})',
                        histtype='step', linewidth=2)
    
    # Customize plot
    plt.xlabel('Strongreject Score', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title('Distribution of Strongreject Scores by Model', fontsize=14, fontweight='bold')
    
    # Apply log scale if requested
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Density (log scale)', fontsize=12, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Spectrogram saved as: {output_file}")

def find_csv_files(root_dir, filename="scored_all_harmful_prompts_cot5_out5.csv"):
    """Find all CSV files with the specified name in the directory structure."""
    root_path = Path(root_dir)
    csv_files = list(root_path.rglob(filename))
    return csv_files

def main():
    parser = argparse.ArgumentParser(description='Generate strongreject score spectrograms')
    parser.add_argument('input_path', help='Path to CSV file or root directory containing CSV files')
    parser.add_argument('-o', '--output', default='strongreject_spectrogram.png', 
                       help='Output filename for the plot')
    parser.add_argument('--filename', default='scored_all_harmful_prompts_cot5_out5.csv',
                       help='Name of CSV file to look for (when scanning directories)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 8],
                       help='Figure size as width height')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Line transparency (0-1)')
    parser.add_argument('--bandwidth', type=float, default=0.02,
                       help='KDE bandwidth for smoothing')
    parser.add_argument('--log-scale', action='store_true',
                       help='Use log scale for y-axis (density)')
    parser.add_argument('--histogram', action='store_true',
                       help='Use histogram instead of KDE for discrete data')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for histogram (default: 50)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    data_dict = {}
    
    if input_path.is_file() and input_path.suffix == '.csv':
        # Single file mode
        model_name, scores = load_and_process_csv(input_path)
        if model_name and scores is not None:
            data_dict[model_name] = scores
    
    elif input_path.is_dir():
        # Directory scanning mode
        csv_files = find_csv_files(input_path, args.filename)
        
        if not csv_files:
            print(f"No files named '{args.filename}' found in {input_path}")
            return
        
        print(f"Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  {csv_file}")
        
        # Process each CSV file
        for csv_file in csv_files:
            model_name, scores = load_and_process_csv(csv_file)
            if model_name and scores is not None:
                data_dict[model_name] = scores
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    if not data_dict:
        print("No valid data found to plot")
        return
    
    # Create spectrogram
    print(f"\nCreating spectrogram for {len(data_dict)} models...")
    create_spectrogram(data_dict, args.output, 
                      figsize=tuple(args.figsize), 
                      alpha=args.alpha, 
                      bandwidth=args.bandwidth,
                      log_scale=args.log_scale,
                      use_histogram=args.histogram,
                      bins=args.bins)

if __name__ == "__main__":
    # Example usage when run directly
    main()
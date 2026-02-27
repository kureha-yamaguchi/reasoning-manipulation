"""
Visualise the transfer efficacy of the first Chain-of-Thought (CoT) sentence
across up to three models (base + two transfer targets) for a single prompt.

For each CoT sample index the script loads per-rollout StrongREJECT scores
from pre-scored CSV files and renders side-by-side boxplots so that the
resampling distributions of the base model and both transfer models can be
compared at a glance.  A scatter marker overlaid on each CoT position shows
the *output target* score, i.e. the mean score obtained when the full CoT
(not just the first sentence) is used as a prefill on the base model.

Typical usage
-------------
uv run -m utils.heuristic.plot_sentence1_transfer \\
    --prompt_index 0 \\
    --base_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
    --transfer_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
    --transfer_model_2 deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \\
    --repetitions 15

Output
------
A PNG figure saved to:
    figures/scored_transfer_results_idx<N>_cot*_transfer_from_<base>.png
"""
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
import statistics 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multiple output rollouts per prompt for non-reasoning models"
    )
    parser.add_argument("--prompt_index", type=int, required=True,
                        help="Original prompt index")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="First CoT sentence taken from this model")
    parser.add_argument("--transfer_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="Prefill attack applied on this model")
    parser.add_argument("--transfer_model_2", type=str, default=None,
                        help="Prefill attack also applied on this model")
    parser.add_argument("--results_dir", type=str, default='results/',
                        help="Results directory")
    parser.add_argument("--repetitions", type=int, default=15,
                        help="Number of output variations per prompt")
    return parser.parse_args()

def load_scored_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Load CSV data with pre-computed scores.
    
    Args:
        csv_path: Path to the CSV file containing scores
    
    Returns:
        List of dictionaries containing all CSV data including scores
    """
    print(f"Loading scored data from: {csv_path}")
    
    # First pass: count rows for progress bar
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_rows = sum(1 for _ in reader)
    
    # Second pass: load data with progress bar
    all_rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_rows, desc="Loading scored data"):
            all_rows.append(row)
    
    print(f"Loaded {len(all_rows)} rows")
    return all_rows

def compute_stats_per_prompt_cot(
    scored_rows: List[Dict[str, str]]
) -> Tuple[List[List[float]], List[List[float]], List[List[str]]]:
    """
    Groups by (prompt, cot_rep_n), then nests results by prompt.
    
    Returns:
        Tuple of (means_per_prompt, std_devs_per_prompt, cots_per_prompt) where each is a 
        list of lists: [[values for prompt1's CoTs], [values for prompt2's CoTs], ...]
    """
    # Group by (prompt, cot_rep_n) to find all outputs for each CoT
    cot_groups = defaultdict(list)
    for row in tqdm(scored_rows, desc="Grouping by (prompt, cot_rep_n)"):
        key = (row['prompt'], row['cot_rep_n'])
        cot_groups[key].append(row)

    # Compute stats per (prompt, cot), organized by prompt
    # The lambda function is called: lambda: {'means': [], 'std_devs': [], 'cots': []}
    # It returns a fresh dictionary: {'means': [], 'std_devs': [], 'cots': []}
    # That dictionary is stored at prompt_to_stats['some_prompt']
    prompt_to_stats = defaultdict(lambda: {'means': []})
    
    for (prompt, cot_rep_n), rows in tqdm(cot_groups.items(), desc="Processing CoT groups"):
        chunk_scores = [float(row['strongreject_score']) for row in rows]
        mean = statistics.mean(chunk_scores)
        prompt_to_stats[prompt]['means'].append(mean)

    # Convert to lists of lists (preserving prompt order if needed)
    means = [stats['means'] for stats in prompt_to_stats.values()]
    
    return means

def main():
    base_model_short = args.base_model.split("/")[-1]
    transfer_model_short = args.transfer_model.split("/")[-1]
    base_csv = f'scored_resampling_results_idx{args.prompt_index}_cot*.csv'
    transfer_csv = f'scored_deepseek_transfer_results_idx{args.prompt_index}_cot*_transfer_from_{base_model_short}.csv'

    pattern_base = os.path.join(
        args.results_dir,
        args.base_model,
        'dataset',
        'resample',
        base_csv
    )
    pattern_transfer = os.path.join(
        args.results_dir,
        args.transfer_model,
        'dataset',
        'resample',
        transfer_csv
    )
    pattern_transfer2 = os.path.join(
        args.results_dir,
        args.transfer_model_2,
        'dataset',
        'resample',
        transfer_csv
    )

    input_paths_base = sorted(glob.glob(pattern_base))
    input_paths_transfer = sorted(glob.glob(pattern_transfer))
    input_paths_transfer2 = sorted(glob.glob(pattern_transfer2))
    
    cot_numbers = [i+1 for i,_ in enumerate(input_paths_base)]
    print('cot_numbers', cot_numbers)

    if not input_paths_base:
        print(f"No input files found matching pattern: {pattern_base}")
        return
    if not input_paths_transfer:
        print(f"No input files found matching pattern: {pattern_transfer}")
        return
    if not input_paths_transfer2:
        print(f"No input files found matching pattern: {pattern_transfer2}")
        return

    print(f"Found {len(input_paths_base)} input file(s): {[os.path.basename(p) for p in input_paths_base]}")
    print(f"Found {len(input_paths_transfer)} input file(s): {[os.path.basename(p) for p in input_paths_transfer]}")
    print(f"Found {len(input_paths_transfer2)} input file(s): {[os.path.basename(p) for p in input_paths_transfer2]}")

    scores_all_base = []
    for input_path in input_paths_base:
        df = pd.read_csv(input_path)
        prompt = df['prompt'][0]
        scores_all_base.append(df['strongreject_score'].values)

    scores_all_transfer = []
    for input_path in input_paths_transfer:
        df = pd.read_csv(input_path)
        scores_all_transfer.append(df['strongreject_score'].values)

    scores_all_transfer2 = []
    for input_path in input_paths_transfer2:
        df = pd.read_csv(input_path)
        scores_all_transfer2.append(df['strongreject_score'].values)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define offset and width for side-by-side boxplots
    offset = 0.25
    width = 0.22

    # Positions for the three sets of boxplots
    positions_base = [x - offset for x in cot_numbers]
    positions_transfer = [x for x in cot_numbers]
    positions_transfer2 = [x + offset for x in cot_numbers]

    # Target output score after full CoT conditioning (not just the first sentence)
    scored_csv = "scored_train_harmful_prompts_cot5_out5.csv"
    scored_csv_path = os.path.join(args.results_dir, args.base_model, "dataset", scored_csv)
    scored_rows = load_scored_csv(scored_csv_path)
    means = compute_stats_per_prompt_cot(scored_rows)
    output_target = means[args.prompt_index]


    # Boxplot for base model (left side)
    bp_base = ax.boxplot(scores_all_base, positions=positions_base, widths=width,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor='lightgreen'),
                        medianprops=dict(color='black'))

    # Boxplot for transfer model (center)
    bp = ax.boxplot(scores_all_transfer, positions=positions_transfer, widths=width,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='black'))

    # Boxplot for transfer model 2 (right side)
    transfer_model_2_short = args.transfer_model_2.split("/")[-1] if args.transfer_model_2 else "transfer_model_2"
    bp2 = ax.boxplot(scores_all_transfer2, positions=positions_transfer2, widths=width,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor='lightsalmon'),
                     medianprops=dict(color='black'))

    # Red markers for output_target (no line)
    ax.scatter(cot_numbers, output_target, c='red', s=80, marker='s', 
            label='Output Target', zorder=5)

    ax.set_xlabel('CoT Number', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f"Transfer Efficacy of the First CoT Sentences from {base_model_short} to {transfer_model_short} \n Prompt={prompt[:60]} \n (Across {args.repetitions} rollouts)", fontsize=12)
    ax.set_xticks(cot_numbers)
    ax.set_xticklabels(cot_numbers)
    ax.legend([bp_base["boxes"][0], bp["boxes"][0], bp2["boxes"][0], ax.collections[0]], 
            [f'Resampling Distribution from {base_model_short}', 
            f'Resampling Distribution from {transfer_model_short}',
            f'Resampling Distribution from {transfer_model_2_short}',
            f'Output Target from {base_model_short}'])
    ax.grid(True, alpha=0.3)
    ax.annotate('(comply)', xy=(1.01, 1), xycoords='axes fraction', fontsize=10, fontstyle='italic', va='top')
    ax.annotate('(refuse)', xy=(1.01, 0), xycoords='axes fraction', fontsize=10, fontstyle='italic', va='bottom')

    plt.tight_layout()
    save_path = os.path.join('figures', f'{os.path.splitext(transfer_csv)[0]}.png')
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    main()
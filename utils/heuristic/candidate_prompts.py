'''
Candidate prompt pool for a new model's quadrant selection: the union of prompts that are
quadrant-eligible (find_quadrant, same thresholds as resample_quadrants) for any reference
model. Writes filtered copies of the three base prompt CSVs (original order, original
basenames) so utils.batch_generation_cot_output can run the usual cot5_out5 baseline on the
pool only, plus candidates.json recording which reference models each prompt came from.

Prompt indices from find_quadrant are positions in the concatenated scored CSVs, so a model
baselined on this pool gets its own indices; match prompts across models by text.

uv run -m utils.heuristic.candidate_prompts --out_dir results/QuixiAI/DeepSeek-R1-AWQ/dataset/candidates
'''

import argparse
import json
import os

import pandas as pd

from utils.heuristic.resample_quadrants import find_quadrant, load_scored_csv

BASE_CSVS = ['dataset/train_harmful_prompts.csv', 'dataset/base/orbench_extra_prompts.csv',
             'dataset/test_harmful_prompts.csv']
SCORED_CSVS = ['scored_train_harmful_prompts_cot5_out5.csv', 'scored_orbench_extra_prompts_cot5_out5.csv',
               'scored_test_harmful_prompts_cot5_out5.csv']
REFERENCE_MODELS = ['deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                    'openai/gpt-oss-20b']


def eligible_prompts(model_name, results_dir, x_threshold, y_threshold):
    rows = load_scored_csv([os.path.join(results_dir, model_name, 'dataset', f) for f in SCORED_CSVS])
    return [p['prompt'] for p in find_quadrant(rows, x_threshold=x_threshold, y_threshold=y_threshold)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model_names', nargs='+', default=REFERENCE_MODELS)
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--x_threshold', type=float, default=0.03)
    parser.add_argument('--y_threshold', type=float, default=0.43)
    parser.add_argument('--limit', type=int, default=None, help='Keep only the first N pool prompts per base CSV (smoke tests)')
    args = parser.parse_args()

    sources = {}
    for model in args.model_names:
        for prompt in eligible_prompts(model, args.results_dir, args.x_threshold, args.y_threshold):
            sources.setdefault(prompt, []).append(model)

    os.makedirs(args.out_dir, exist_ok=True)
    kept, missing = [], set(sources)
    for base_csv in BASE_CSVS:
        df = pd.read_csv(base_csv)
        df = df[df['prompt'].isin(sources)]
        if args.limit is not None:
            df = df.head(args.limit)
        kept += df['prompt'].tolist()
        missing -= set(df['prompt'])
        df.to_csv(os.path.join(args.out_dir, os.path.basename(base_csv)), index=False)
    if missing and args.limit is None:
        raise ValueError(f'{len(missing)} eligible prompts not found in base CSVs')

    with open(os.path.join(args.out_dir, 'candidates.json'), 'w') as f:
        json.dump({'args': vars(args), 'n_pool': len(kept),
                   'n_eligible_per_model': {m: sum(m in s for s in sources.values()) for m in args.model_names},
                   'prompts': [{'prompt': p, 'eligible_for': sources[p]} for p in kept]}, f, indent=2)
    print(f'Pool: {len(kept)} prompts -> {args.out_dir}')


if __name__ == '__main__':
    main()

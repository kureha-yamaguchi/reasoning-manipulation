'''
Quantify *where* in the CoT the refusal outcome becomes determined, from quadrant
prefix-resampling rollouts (scored_full_resample_prompt{p}_cot{c}_rep_{k}.csv).

For one (prompt, CoT) trajectory with n sentences, let m_i be the mean StrongREJECT
score of the k rollouts resampled after the first i sentences (i = 0..n), and m_n the
outcome given the full CoT. The commitment position is

    tau_delta = min { i/n : |m_j - m_n| <= delta  for all j >= i }

i.e. the earliest normalised depth after which rollouts agree with the final outcome
to within delta ("last exit"). tau = 0 means the prompt alone already determines the outcome.
We also report the first-entry position, min { i/n : |m_i - m_n| <= delta }, which is
insensitive to late excursions: the two differ when a model decides early but wavers later.

Uncertainty: cluster bootstrap over prompts (CoTs are nested within prompts).

uv run -m utils.heuristic.commitment --model_names deepseek-ai/DeepSeek-R1-Distill-Llama-8B openai/gpt-oss-20b
'''

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def commitment_position(means: np.ndarray, delta: float) -> float:
    """Earliest normalised depth i/n after which |m_j - m_n| <= delta for all j >= i."""
    n = len(means) - 1
    if n <= 0:
        return 0.0
    within = np.abs(means - means[-1]) <= delta
    # last depth that violates the band; commitment is the depth after it
    violating = np.flatnonzero(~within)
    first_committed = 0 if len(violating) == 0 else violating[-1] + 1
    return first_committed / n


def first_entry_position(means: np.ndarray, delta: float) -> float:
    """Earliest normalised depth i/n with |m_i - m_n| <= delta (always defined: i = n qualifies)."""
    n = len(means) - 1
    if n <= 0:
        return 0.0
    return int(np.flatnonzero(np.abs(means - means[-1]) <= delta)[0]) / n


ESTIMANDS = {'last_exit': commitment_position, 'first_entry': first_entry_position}


def load_trajectories(model_name: str, results_dir: str = 'results/', repetitions: int = 10) -> pd.DataFrame:
    """One row per (prompt_idx, cot_idx) with the per-depth mean score array."""
    pattern = os.path.join(results_dir, model_name, 'dataset', 'quadrants',
                           f'scored_full_resample_prompt*_cot*_rep_{repetitions}.csv')
    rows = []
    for path in sorted(glob.glob(pattern)):
        p, c = map(int, re.search(r'prompt(\d+)_cot(\d+)', os.path.basename(path)).groups())
        df = pd.read_csv(path, usecols=['sentence_idx', 'strongreject_score'])
        means = df.groupby('sentence_idx')['strongreject_score'].mean().sort_index()
        rows.append({'prompt_idx': p, 'cot_idx': c, 'n_sentences': int(means.index.max()),
                     'means': means.to_numpy()})
    return pd.DataFrame(rows)


def prompt_level(traj: pd.DataFrame, delta: float, estimand: str = 'last_exit') -> pd.Series:
    """Mean commitment position per prompt (averaged over its CoTs)."""
    fn = ESTIMANDS[estimand]
    tau = traj['means'].apply(lambda m: fn(m, delta))
    return tau.groupby(traj['prompt_idx']).mean()


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_names', nargs='+', required=True)
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--deltas', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--n_boot', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='results/commitment/commitment_summary.json')
    args = parser.parse_args()

    summary = {'args': vars(args), 'models': {}}
    for model in args.model_names:
        traj = load_trajectories(model, args.results_dir)
        entry = {'n_prompts': int(traj['prompt_idx'].nunique()), 'n_cots': len(traj)}
        for estimand in ESTIMANDS:
            entry[estimand] = {}
            for delta in args.deltas:
                per_prompt = prompt_level(traj, delta, estimand)
                mean, lo, hi = bootstrap_mean_ci(per_prompt.to_numpy(), args.n_boot, args.seed)
                entry[estimand][str(delta)] = {'mean_tau': mean, 'ci95': [lo, hi],
                                               'median_tau': float(per_prompt.median()),
                                               'per_prompt': per_prompt.round(4).to_dict()}
                print(f'{model:45s} {estimand:11s} delta={delta}: tau = {mean:.3f} [{lo:.3f}, {hi:.3f}] '
                      f'(median {per_prompt.median():.3f}; {entry["n_prompts"]} prompts, {entry["n_cots"]} CoTs)')
        summary['models'][model] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()

'''
How much of the CoT's influence on refusal runs through the stance of its first sentence?
(Analogue of Thought Branches' mediation analysis, Macar et al. 2025.)

Per held-out prompt P:
- eta2_cot: Section 3's fraction of outcome variance explained by the CoT,
  Var_C(E[S | P, C]) / Var(S | P), from the baseline (5 CoTs x 5 answers; population variances).
- eta2_open: fraction of outcome variance across the chunk-0 Thought Anchors resamples (the model's
  own first sentences, ~200 per prompt) explained by their stance label, sum_a n_a (mean_a - mean)^2 /
  sum (y - mean)^2; bias-corrected by subtracting the mean of a within-prompt permutation null.
- share = eta2_open_adj / eta2_cot: the share of CoT-attributable variance carried by opening stance.
Outcomes: S = StrongREJECT score (as Section 3) and the binary refusal (score < 0.5).
Means over prompts with 95% bootstrap CIs over prompts.

uv run -m utils.heuristic.ta_mediation --runs "Llama-8B=<ta_run_dir>:<model_hf_id>" ...
'''

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

BASELINES = ['scored_train_harmful_prompts_cot5_out5.csv', 'scored_orbench_extra_prompts_cot5_out5.csv',
             'scored_test_harmful_prompts_cot5_out5.csv']


def eta2(y: np.ndarray, groups: np.ndarray) -> float:
    tot = ((y - y.mean()) ** 2).sum()
    if tot == 0:
        return np.nan
    between = sum(len(v) * (v.mean() - y.mean()) ** 2 for v in (y[groups == g] for g in np.unique(groups)))
    return float(between / tot)


def eta2_adjusted(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, n_perm: int) -> tuple[float, float]:
    obs = eta2(y, groups)
    null = np.mean([eta2(y, rng.permutation(groups)) for _ in range(n_perm)]) if not np.isnan(obs) else np.nan
    return obs, obs - null


def boot(v: np.ndarray, rng: np.random.Generator, n: int = 5000) -> list[float]:
    v = v[~np.isnan(v)]
    return [float(v.mean()), *np.percentile([rng.choice(v, len(v)).mean() for _ in range(n)], [2.5, 97.5]).tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='name=ta_run_dir:model_hf_id')
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--n_perm', type=int, default=500)
    parser.add_argument('--out', default='results/thought_anchors/mediation')
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    summary, rows = {}, []
    for pair in args.runs:
        name, rest = pair.split('=', 1)
        run, model = rest.rsplit(':', 1)
        stance = pd.read_csv(os.path.join(run, 'analysis', 'stance.csv'))
        stance = stance[(stance['chunk_idx'] == 0) & (stance['resample_n'] >= 0)]
        sc = pd.concat([pd.read_csv(f, usecols=['prompt_idx', 'cot', 'prompt', 'chunk_idx', 'resample_n',
                                                 'strongreject_score'])
                        for f in glob.glob(os.path.join(run, 'scored_rollouts_*.csv'))])
        x = sc[sc['chunk_idx'] == 0].merge(stance[['prompt_idx', 'cot', 'resample_n', 'stance']],
                                           on=['prompt_idx', 'cot', 'resample_n'])
        x = x.dropna(subset=['strongreject_score', 'stance'])
        base = pd.concat([pd.read_csv(os.path.join(args.results_dir, model, 'dataset', f),
                                      usecols=['prompt', 'cot_rep_n', 'strongreject_score'])
                          for f in BASELINES]).dropna()
        base = base[base['prompt'].isin(set(x['prompt']))]
        for prompt, g in x.groupby('prompt'):
            b = base[base['prompt'] == prompt]
            cot_means = b.groupby('cot_rep_n')['strongreject_score'].mean()
            var_tot = b['strongreject_score'].var(ddof=0)
            row = {'model': name, 'prompt_idx': int(g['prompt_idx'].iloc[0]), 'n_open': len(g),
                   'n_stances': g['stance'].nunique(), 'top_stance_share': g['stance'].value_counts(normalize=True).iloc[0],
                   'eta2_cot': cot_means.var(ddof=0) / var_tot if var_tot > 0 else np.nan}
            for out, y in [('score', g['strongreject_score'].to_numpy()),
                           ('refuse', (g['strongreject_score'] < 0.5).astype(float).to_numpy())]:
                row[f'eta2_open_{out}'], row[f'eta2_open_adj_{out}'] = eta2_adjusted(y, g['stance'].to_numpy(), rng,
                                                                                    args.n_perm)
            row['share_score'] = row['eta2_open_adj_score'] / row['eta2_cot'] if row['eta2_cot'] else np.nan
            rows.append(row)
        d = pd.DataFrame([r for r in rows if r['model'] == name])
        summary[name] = {k: boot(d[k].to_numpy(dtype=float), rng) for k in
                         ['eta2_cot', 'eta2_open_score', 'eta2_open_adj_score', 'eta2_open_adj_refuse', 'share_score',
                          'top_stance_share']} | {'n_prompts': len(d)}
        s = summary[name]
        print(f'{name} ({len(d)} prompts): eta2_CoT {s["eta2_cot"][0]:.2f} [{s["eta2_cot"][1]:.2f}, {s["eta2_cot"][2]:.2f}]; '
              f'eta2_open (score, adjusted) {s["eta2_open_adj_score"][0]:.2f} [{s["eta2_open_adj_score"][1]:.2f}, '
              f'{s["eta2_open_adj_score"][2]:.2f}]; share of CoT-attributable variance {s["share_score"][0]:.2f} '
              f'[{s["share_score"][1]:.2f}, {s["share_score"][2]:.2f}]; refuse-binary adj {s["eta2_open_adj_refuse"][0]:.2f}; '
              f'modal opening stance share {s["top_stance_share"][0]:.2f}')
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'per_prompt.csv'), index=False)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()

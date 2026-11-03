'''
Analyse a ta_prefill run (scored with ta_score): P(refuse) (StrongREJECT score < 0.5) per prompt and
condition; frame contrasts per prompt with a sign test and a 95% cluster-bootstrap CI over prompts.
If --resample_run is given (a ta_resample run with stance labels), the matching on-policy contrast
from chunk-0 resamples (P(refuse | opening stance), per prompt) is printed alongside.

Writes <run_dir>/analysis/prefill_by_prompt.csv and prefill_summary.json.

uv run -m utils.heuristic.ta_prefill_analyse --run_dir results/thought_anchors_prefill/M/latest
'''

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CONTRASTS = {  # (frame a, frame b): P(refuse | a) - P(refuse | b)
    'distill': [('requester', 'task'), ('requester', 'baseline'), ('task', 'baseline')],
    'gpt-oss': [('disallowed', 'not_disallowed'), ('disallowed', 'opening_only'),
                ('not_disallowed', 'opening_only'), ('neutral', 'opening_only'), ('opening_only', 'baseline')],
}


def contrast(by_prompt: pd.DataFrame, a: str, b: str, n_boot: int, rng: np.random.Generator) -> dict:
    d = (by_prompt[a] - by_prompt[b]).dropna().to_numpy()
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    n_pos, n_nonzero = int((d > 0).sum()), int((d != 0).sum())
    return {'contrast': f'{a} - {b}', 'n_prompts': len(d), 'mean': float(d.mean()), 'ci_lo': float(lo),
            'ci_hi': float(hi), 'median': float(np.median(d)), 'n_positive': n_pos,
            'sign_test_p': float(binomtest(n_pos, n_nonzero).pvalue) if n_nonzero else np.nan}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--resample_run', default=None)
    parser.add_argument('--n_boot', type=int, default=10000)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.run_dir, 'scored_rollouts_prefill.csv'))
    print(f'{len(df)} rollouts, {df["strongreject_score"].isna().sum()} unscored (dropped)')
    df = df.dropna(subset=['strongreject_score'])
    df['refuse'] = df['strongreject_score'] < 0.5
    by_cond = df.groupby(['prompt_idx', 'condition', 'frame'])['refuse'].agg(['mean', 'size']).reset_index()
    by_frame = df.groupby(['prompt_idx', 'frame'])['refuse'].mean().unstack()  # paraphrases pooled
    family = 'gpt-oss' if 'disallowed' in by_frame else 'distill'

    print('\nP(refuse) by condition, mean over prompts [min, max]:')
    for (cond, frame), g in by_cond.groupby(['condition', 'frame']):
        print(f'  {frame:15s} {cond:18s} {g["mean"].mean():.2f} [{g["mean"].min():.2f}, {g["mean"].max():.2f}]')

    rng = np.random.default_rng(0)
    results = [contrast(by_frame, a, b, args.n_boot, rng) for a, b in CONTRASTS[family] if a in by_frame and b in by_frame]
    print('\nPer-prompt contrasts (paraphrases pooled):')
    for r in results:
        print(f'  {r["contrast"]:30s} mean {r["mean"]:+.2f} [{r["ci_lo"]:+.2f}, {r["ci_hi"]:+.2f}] '
              f'positive {r["n_positive"]}/{r["n_prompts"]}, sign test p={r["sign_test_p"]:.1e}')

    if args.resample_run:
        st = pd.read_csv(os.path.join(args.resample_run, 'analysis', 'stance.csv'))
        sc = pd.concat([pd.read_csv(f, usecols=['prompt_idx', 'cot', 'chunk_idx', 'resample_n', 'strongreject_score'])
                        for f in glob.glob(os.path.join(args.resample_run, 'scored_rollouts_*.csv'))])
        x = sc[sc['chunk_idx'] == 0].merge(st[st['chunk_idx'] == 0], on=['prompt_idx', 'cot', 'chunk_idx', 'resample_n'])
        x = x.dropna(subset=['stance', 'strongreject_score'])
        on_policy = (x.assign(refuse=x['strongreject_score'] < 0.5)
                     .groupby(['prompt_idx', 'stance'])['refuse'].mean().unstack())
        if {'requester', 'task'} <= set(on_policy) and family == 'distill':
            both = pd.DataFrame({'prefill': by_frame['requester'] - by_frame['task'],
                                 'resample': on_policy['requester'] - on_policy['task']}).dropna()
            print(f'\nrequester - task, prefill vs on-policy resampling (prompts with both): n={len(both)}, '
                  f'means {both["prefill"].mean():+.2f} vs {both["resample"].mean():+.2f}, '
                  f'Pearson r={both.corr().iloc[0, 1]:.2f}')

    out_dir = os.path.join(args.run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    by_cond.to_csv(os.path.join(out_dir, 'prefill_by_prompt.csv'), index=False)
    with open(os.path.join(out_dir, 'prefill_summary.json'), 'w') as f:
        json.dump({'family': family, 'n_rollouts': len(df), 'contrasts': results}, f, indent=2)
    print(f'\nSaved {out_dir}/prefill_by_prompt.csv and prefill_summary.json')


if __name__ == '__main__':
    main()

'''
Where does the refusal decision lock in? Per model, from ta_analyse outputs (stance criterion,
outcome = StrongREJECT score):

- Cumulative importance: share of a CoT's total counterfactual importance (KL) carried by chunks at
  relative position <= x (x = chunk_idx / (n_chunks - 1)); averaged over CoTs, 95% cluster bootstrap
  over prompts.
- Lock-in point: first chunk k such that |P(refuse)_j - P(refuse)_last| <= eps for every j >= k (eps
  0.1; 0.05 and 0.2 as sensitivity), as an index and as a fraction of the CoT.
- Load-bearing sentences: CF >= 0.1 nats with the 95% CI of delta excluding 0; their positions.

Writes results/thought_anchors/lockin/summary.json and figures/ta_lockin.{pdf,png}.

uv run -m utils.heuristic.ta_lockin --runs "Llama-8B=<run_dir>" ...
'''

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.heuristic.plot_std_overlay import MODEL_COLOURS

GRID = np.linspace(0, 1, 101)


def lock_in(p: np.ndarray, eps: float) -> int:
    """First index k with |p_j - p_last| <= eps for all j >= k."""
    off = np.abs(p - p[-1]) > eps
    return int(np.flatnonzero(off).max() + 1) if off.any() else 0


def per_cot(res: pd.DataFrame, eps_list: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for (p, c), g in res.groupby(['prompt_idx', 'cot']):
        g = g.sort_values('chunk_idx')
        n = len(g)
        x = g['chunk_idx'].to_numpy() / max(n - 1, 1)
        cf = g['counterfactual_importance'].fillna(0).to_numpy()
        tot = cf.sum()
        cum = np.array([cf[x <= t].sum() / tot for t in GRID]) if tot > 0 else np.full(len(GRID), np.nan)
        curves.append(pd.DataFrame({'prompt_idx': p, 'cot': c, 'x': GRID, 'cum': cum}))
        pr = g['p_target'].to_numpy()
        row = {'prompt_idx': p, 'cot': c, 'label': g['cot_label'].iloc[0], 'n_chunks': n}
        for eps in eps_list:
            k = lock_in(pr, eps)
            row[f'lockin_{eps}'], row[f'lockin_frac_{eps}'] = k, k / n
        lb = g[(g['counterfactual_importance'] >= 0.1)
               & ((g['counterfactual_delta_lo'] > 0) | (g['counterfactual_delta_hi'] < 0))]
        row['n_load_bearing'] = len(lb)
        row['lb_idx'] = lb['chunk_idx'].tolist()
        row['lb_frac'] = (lb['chunk_idx'] / max(n - 1, 1)).round(3).tolist()
        row['share_cf_chunk0'] = cf[0] / tot if tot > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def boot_mean(df: pd.DataFrame, col: str, rng: np.random.Generator, n_boot: int, stat=np.mean) -> list[float]:
    prompts = df['prompt_idx'].unique()
    by = {p: df.loc[df['prompt_idx'] == p, col].dropna().to_numpy() for p in prompts}
    est = float(stat(np.concatenate(list(by.values()))))
    boots = [stat(np.concatenate([by[p] for p in rng.choice(prompts, len(prompts))])) for _ in range(n_boot)]
    return [est, *np.percentile(boots, [2.5, 97.5]).tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='label=run_dir (label as in MODEL_COLOURS)')
    parser.add_argument('--eps', type=float, nargs='+', default=[0.1, 0.05, 0.2])
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--out', default='results/thought_anchors/lockin')
    parser.add_argument('--fig', default='figures/ta_lockin.pdf')
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    summary, cots_all, curves_all = {}, [], []
    for pair in args.runs:
        name, run = pair.split('=', 1)
        res = pd.read_csv(os.path.join(run, 'analysis', 'per_chunk_stance.csv'))
        res = res[res['outcome'] == 'score']
        cots, curves = per_cot(res, args.eps)
        cots_all.append(cots.assign(model=name))
        curves_all.append(curves.assign(model=name))
        lb_frac = np.concatenate([np.array(f) for f in cots['lb_frac']]) if cots['n_load_bearing'].sum() else np.array([])
        s = {'run': run, 'n_cots': len(cots), 'n_prompts': int(cots['prompt_idx'].nunique()),
             'median_chunks': float(cots['n_chunks'].median()),
             'share_cf_chunk0': boot_mean(cots, 'share_cf_chunk0', rng, args.n_boot),
             'n_load_bearing_per_cot': boot_mean(cots, 'n_load_bearing', rng, args.n_boot),
             'load_bearing_frac_quartiles': np.percentile(lb_frac, [25, 50, 75]).round(3).tolist() if len(lb_frac) else None,
             'load_bearing_at_chunk0': float(np.mean([0 in i for i in cots['lb_idx']]))}
        for eps in args.eps:
            s[f'lockin_idx_median_eps{eps}'] = boot_mean(cots, f'lockin_{eps}', rng, args.n_boot, np.median)
            s[f'lockin_frac_mean_eps{eps}'] = boot_mean(cots, f'lockin_frac_{eps}', rng, args.n_boot)
        for lab in ['refusing', 'complying']:
            sub = cots[cots['label'] == lab]
            s[f'lockin_idx_median_eps0.1_{lab}'] = boot_mean(sub, 'lockin_0.1', rng, args.n_boot, np.median)
        summary[name] = s
        print(f'{name}: {json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in s.items() if k != "run"})}')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    cots = pd.concat(cots_all, ignore_index=True)
    cots.to_csv(os.path.join(args.out, 'per_cot.csv'), index=False)
    curves = pd.concat(curves_all, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={'width_ratios': [1.25, 1, 1]})
    names = list(summary)
    for name in names:
        col = MODEL_COLOURS.get(summary[name]['run'].split('results/thought_anchors/')[-1].rsplit('/', 1)[0], None)
        cv = curves[curves['model'] == name].dropna()
        mean = cv.groupby('x')['cum'].mean()
        prompts = cv['prompt_idx'].unique()
        boots = np.array([pd.concat([cv[cv['prompt_idx'] == p] for p in rng.choice(prompts, len(prompts))])
                          .groupby('x')['cum'].mean().to_numpy() for _ in range(300)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        axes[0].fill_between(mean.index, lo, hi, color=col, alpha=0.18, linewidth=0)
        axes[0].plot(mean.index, mean.to_numpy(), color=col, linewidth=2, label=name)
    axes[0].set_xlabel('relative position in CoT (0 = first sentence)')
    axes[0].set_ylabel('cumulative share of counterfactual\nimportance (mean over CoTs)')
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    for ax, col_name, xlabel in [(axes[1], 'lockin_0.1', 'lock-in sentence index (ε = 0.1)'),
                                 (axes[2], 'lockin_frac_0.1', 'lock-in, fraction of CoT (ε = 0.1)')]:
        for y, name in enumerate(names):
            col = MODEL_COLOURS.get(summary[name]['run'].split('results/thought_anchors/')[-1].rsplit('/', 1)[0], None)
            v = cots.loc[cots['model'] == name, col_name].to_numpy()
            jit = rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(v, y + jit, s=14, color=col, alpha=0.6, linewidths=0)
            med = np.median(v)
            ax.plot([med, med], [y - 0.3, y + 0.3], color=col, linewidth=2.5)
        ax.set_yticks(range(len(names)), names if ax is axes[1] else [''] * len(names), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
    axes[1].set_xlim(left=-1)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig)
    fig.savefig(args.fig.replace('.pdf', '.png'), dpi=150)
    print(f'Saved {args.fig} and {args.out}/summary.json')


if __name__ == '__main__':
    main()

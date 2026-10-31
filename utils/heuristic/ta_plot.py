'''
Figures for a ta_analyse run (reads <run_dir>/analysis/per_chunk.csv and, if present, taxonomy.csv).

1. cot_panels.pdf — one column per CoT: P(refuse) after resampling from each chunk (95% Wilson
   band), and below it the signed counterfactual effect of each chunk,
   delta = P(refuse | kept) − P(refuse | counterfactual), with 95% binomial-bootstrap whiskers.
2. category_importance.pdf — counterfactual and counterfactual++ importance by taxonomy
   category: mean over chunks with 95% cluster-bootstrap CI (resampling prompts).

uv run -m utils.heuristic.ta_plot --run_dir results/thought_anchors/M/latest
'''

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOWARDS_REFUSAL, TOWARDS_COMPLIANCE = '#2166AC', '#B2182B'  # ColorBrewer RdBu poles
INK, MUTED = '#222222', '#888888'


def wilson(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def cot_panels(res: pd.DataFrame, path: str):
    cots = res.groupby(['prompt_idx', 'cot'], sort=True)
    fig, axes = plt.subplots(2, len(cots), figsize=(3.2 * len(cots), 4.6), sharey='row',
                             gridspec_kw={'height_ratios': [1, 1.2]}, squeeze=False)
    for col, ((p, c), g) in enumerate(cots):
        g = g.sort_values('chunk_idx')
        x = g['chunk_idx'].to_numpy()
        lo, hi = wilson(g['p_target'].to_numpy(), g['n'].to_numpy())
        ax = axes[0, col]
        ax.fill_between(x, lo, hi, color=MUTED, alpha=0.25, linewidth=0)
        ax.plot(x, g['p_target'], color=INK, linewidth=2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f'prompt {p}, CoT {c} ({g["cot_label"].iloc[0]})', fontsize=9)
        ax = axes[1, col]
        d = g['counterfactual_delta'].to_numpy()
        colours = np.where(d >= 0, TOWARDS_REFUSAL, TOWARDS_COMPLIANCE)
        err = np.vstack([d - g['counterfactual_delta_lo'], g['counterfactual_delta_hi'] - d])
        ax.bar(x, d, color=colours, width=0.8)
        ax.errorbar(x, d, yerr=np.nan_to_num(err), fmt='none', ecolor=MUTED, elinewidth=0.8)
        ax.axhline(0, color=INK, linewidth=0.6)
        ax.set_xlabel('chunk index')
    axes[0, 0].set_ylabel('P(refuse) after\nresampling from chunk')
    axes[1, 0].set_ylabel('counterfactual effect\nΔP(refuse) of chunk')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TOWARDS_REFUSAL), plt.Rectangle((0, 0), 1, 1, color=TOWARDS_COMPLIANCE)]
    fig.legend(handles, ['chunk pushes towards refusal', 'chunk pushes towards compliance'], loc='lower center',
               ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)


def cluster_ci(df: pd.DataFrame, col: str, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    prompts = df['prompt_idx'].unique()
    by = {p: df.loc[df['prompt_idx'] == p, col].to_numpy() for p in prompts}
    boots = [np.mean(np.concatenate([by[p] for p in rng.choice(prompts, len(prompts))])) for _ in range(n_boot)]
    return float(df[col].mean()), *np.percentile(boots, [2.5, 97.5])


def category_importance(res: pd.DataFrame, tax: pd.DataFrame, path: str, n_boot: int = 2000):
    df = res.merge(tax[['prompt_idx', 'cot', 'chunk_idx', 'category']], on=['prompt_idx', 'cot', 'chunk_idx'])
    df = df[df['chunk_idx'] < df['n_chunks'] - 1]  # the last chunk scores 0 by definition
    rng = np.random.default_rng(0)
    metrics = [('counterfactual_importance', 'counterfactual'), ('counterfactual_pp_importance', 'counterfactual++')]
    order = df.groupby('category')['counterfactual_importance'].mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.35 * len(order) + 1.2), sharey=True)
    for ax, (col, name) in zip(axes, metrics):
        for y, cat in enumerate(order):
            sub = df[df['category'] == cat]
            m, lo, hi = cluster_ci(sub, col, n_boot, rng)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt='o', color=INK, ecolor=MUTED, markersize=5)
            ax.text(hi, y, f'  n={len(sub)}', va='center', fontsize=7, color=MUTED)
        ax.set_xlabel(f'{name} importance (KL, nats)')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(order)), [c.replace('_', ' ') for c in order], fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--outcome', default='score')
    args = parser.parse_args()
    out_dir = os.path.join(args.run_dir, 'analysis')
    res = pd.read_csv(os.path.join(out_dir, 'per_chunk.csv'))
    res = res[res['outcome'] == args.outcome]
    cot_panels(res, os.path.join(out_dir, 'cot_panels.pdf'))
    tax_path = os.path.join(out_dir, 'taxonomy.csv')
    if os.path.exists(tax_path):
        category_importance(res, pd.read_csv(tax_path), os.path.join(out_dir, 'category_importance.pdf'))
    print(f'Saved figures to {out_dir}')


if __name__ == '__main__':
    main()

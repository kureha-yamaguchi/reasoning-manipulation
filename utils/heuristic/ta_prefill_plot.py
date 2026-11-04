'''
Figure for the prefill (sentence-forcing) experiment, all models: per model, P(refuse) under each forced
opening (one dot per prompt, mean with 95% bootstrap CI over prompts); final panel, per prompt, the forced
requester − task effect against the same contrast from the model's own chunk-0 resamples (Thought Anchors
run, stance labels), for the models with requester/task conditions.

uv run -m utils.heuristic.ta_prefill_plot --runs "R1-Distill-Llama-8B=<prefill_run>:<ta_run>" ...
'''

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.heuristic.plot_std_overlay import MODEL_COLOURS

REFUSE, COMPLY, NEUTRAL = '#2166AC', '#B2182B', '#7c8691'
FRAME_COLOUR = {'baseline': NEUTRAL, 'requester': REFUSE, 'task': COMPLY, 'opening_only': NEUTRAL,
                'neutral': '#a9b1b9', 'disallowed': REFUSE, 'not_disallowed': COMPLY}
ORDER = ['none', 'req_user_asking', 'req_got_request', 'req_someone_wants', 'task_need_figure', 'task_trying',
         'task_want', 'user_asks', 'check_policy', 'disallowed', 'not_disallowed']
LABEL = {'none': 'no prefill', 'req_user_asking': '"the user is asking me"', 'req_got_request': '"I just got a request"',
         'req_someone_wants': '"someone wants me to"', 'task_need_figure': '"I need to figure out how to"',
         'task_trying': '"I\'m trying to"', 'task_want': '"I want to"', 'user_asks': '"The user asks: …"',
         'check_policy': '+ "We need to check policy."', 'disallowed': '+ "This is disallowed content."',
         'not_disallowed': '+ "This is not disallowed content."'}


def model_key(path: str) -> str:
    return path.split('results/thought_anchors_prefill/')[-1].rsplit('/', 1)[0]


def on_policy(ta_run: str) -> pd.Series:
    st = pd.read_csv(os.path.join(ta_run, 'analysis', 'stance.csv'))
    sc = pd.concat([pd.read_csv(f, usecols=['prompt_idx', 'cot', 'chunk_idx', 'resample_n', 'strongreject_score'])
                    for f in glob.glob(os.path.join(ta_run, 'scored_rollouts_*.csv'))])
    x = sc[sc['chunk_idx'] == 0].merge(st[st['chunk_idx'] == 0], on=['prompt_idx', 'cot', 'chunk_idx', 'resample_n'])
    x = x.dropna(subset=['stance', 'strongreject_score'])
    p = x.assign(refuse=x['strongreject_score'] < 0.5).groupby(['prompt_idx', 'stance'])['refuse'].agg(['mean', 'size'])
    p = p[p['size'] >= 5]['mean'].unstack()
    return (p['requester'] - p['task']).dropna() if {'requester', 'task'} <= set(p) else pd.Series(dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='name=prefill_run:ta_run')
    parser.add_argument('--fig', default='figures/ta_prefill.pdf')
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    runs = [(n, *r.split(':')) for n, r in (p.split('=', 1) for p in args.runs)]

    fig, axes = plt.subplots(1, len(runs) + 1, figsize=(3.5 * (len(runs) + 1), 4.6),
                             gridspec_kw={'width_ratios': [1] * len(runs) + [1.1]})
    scatter = []
    for ax, (name, prun, trun) in zip(axes, runs):
        d = pd.read_csv(os.path.join(prun, 'scored_rollouts_prefill.csv'), usecols=['prompt_idx', 'condition', 'frame',
                                                                                    'strongreject_score']).dropna()
        d['refuse'] = d['strongreject_score'] < 0.5
        by = d.groupby(['condition', 'frame', 'prompt_idx'])['refuse'].mean().reset_index()
        conds = [c for c in ORDER if c in set(by['condition'])]
        for y, c in enumerate(conds):
            v = by.loc[by['condition'] == c, 'refuse'].to_numpy()
            col = FRAME_COLOUR[by.loc[by['condition'] == c, 'frame'].iloc[0]]
            ax.scatter(v, y + rng.uniform(-0.15, 0.15, len(v)), s=10, color=col, alpha=0.45, linewidths=0)
            boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ax.plot([lo, hi], [y, y], color=col, linewidth=2.2)
            ax.plot(v.mean(), y, 'o', color=col, markersize=6, markeredgecolor='white', markeredgewidth=0.8)
        ax.set_yticks(range(len(conds)), [LABEL[c] for c in conds], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel('P(refuse)')
        ax.set_title(name, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        frames = d.groupby(['prompt_idx', 'frame'])['refuse'].mean().unstack()
        if {'requester', 'task'} <= set(frames):
            forced = (frames['requester'] - frames['task']).rename('forced')
            both = pd.concat([forced, on_policy(trun).rename('natural')], axis=1).dropna()
            scatter.append((name, both))
    ax = axes[-1]
    for name, both in scatter:
        col = MODEL_COLOURS.get(model_key(next(p for n, p, _ in runs if n == name)), NEUTRAL)
        r = both.corr().iloc[0, 1]
        ax.scatter(both['natural'], both['forced'], s=18, color=col, alpha=0.8, linewidths=0,
                   label=f'{name.replace("R1-Distill-", "").replace(" (AWQ)", "")}: r = {r:.2f}')
    ax.plot([-0.2, 1], [-0.2, 1], color=NEUTRAL, linewidth=0.8, linestyle='--')
    ax.set_xlim(-0.25, 1.05)
    ax.set_ylim(-0.25, 1.05)
    ax.axhline(0, color=NEUTRAL, linewidth=0.5)
    ax.axvline(0, color=NEUTRAL, linewidth=0.5)
    ax.set_xlabel('natural openings:\nΔP(refuse), requester − task')
    ax.set_ylabel('forced openings:\nΔP(refuse), requester − task')
    ax.set_title('forced vs natural (per prompt)', fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc='lower right', handletextpad=0.2, borderaxespad=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, bbox_inches='tight')
    fig.savefig(args.fig.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'Saved {args.fig}')


if __name__ == '__main__':
    main()

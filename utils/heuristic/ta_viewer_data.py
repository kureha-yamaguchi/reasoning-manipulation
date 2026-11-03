'''
Compact JSON of Thought Anchors runs for the interactive trace viewer: per CoT, every original
chunk (sentence, stance, P(refuse) after resampling, counterfactual effect under the stance and
cosine criteria) and its resamples grouped by stance (counts, refusals, a few verbatim examples).
Answers are included only as short snippets for chunk-0 examples.

uv run -m utils.heuristic.ta_viewer_data --runs Llama-8B=results/thought_anchors/.../<run> ... --out viewer.json \
    --html ta_viewer.html   # fills viewer/ta_viewer_template.html with the data
'''

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

T_CHARS, ANSWER_CHARS = 220, 240


def r3(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def run_data(run_dir: str) -> list[dict]:
    a = os.path.join(run_dir, 'analysis')
    st = pd.read_csv(os.path.join(a, 'per_chunk_stance.csv'))
    cos = pd.read_csv(os.path.join(a, 'per_chunk.csv'))
    st, cos = st[st['outcome'] == 'score'], cos[cos['outcome'] == 'score']
    labels = pd.read_csv(os.path.join(a, 'stance.csv'))
    s_lab = labels[labels['resample_n'] == -1].set_index(['prompt_idx', 'cot', 'chunk_idx'])['stance']
    t_lab = labels[labels['resample_n'] >= 0].set_index(['prompt_idx', 'cot', 'chunk_idx', 'resample_n'])['stance']
    cots = []
    for path in sorted(glob.glob(os.path.join(run_dir, 'scored_rollouts_*.csv'))):
        df = pd.read_csv(path, usecols=['prompt_idx', 'cot', 'cot_label', 'prompt', 'chunk_idx', 'resample_n',
                                        'chunk_removed', 'chunk_resampled', 'output', 'strongreject_score'],
                         keep_default_na=False, na_values=[''])
        df = df.dropna(subset=['strongreject_score'])
        p, c = int(df['prompt_idx'].iloc[0]), int(df['cot'].iloc[0])
        df['refuse'] = (df['strongreject_score'] < 0.5).astype(int)
        df['stance'] = t_lab.reindex(pd.MultiIndex.from_frame(df[['prompt_idx', 'cot', 'chunk_idx', 'resample_n']])).to_numpy()
        df['stance'] = df['stance'].fillna('unlabelled')
        sg = st[(st['prompt_idx'] == p) & (st['cot'] == c)].set_index('chunk_idx')
        cg = cos[(cos['prompt_idx'] == p) & (cos['cot'] == c)].set_index('chunk_idx')
        chunks = []
        for i, g in df.groupby('chunk_idx'):
            k = 6 if i <= 2 else 2
            groups = []
            for stance, h in g.groupby('stance'):
                ex = h.drop_duplicates('chunk_resampled').head(k)
                groups.append({'s': stance, 'n': len(h), 'r': int(h['refuse'].sum()),
                               'ex': [[str(t)[:T_CHARS], int(rf)] + ([str(o).strip()[:ANSWER_CHARS]] if i == 0 else [])
                                      for t, rf, o in zip(ex['chunk_resampled'], ex['refuse'], ex['output'])]})
            groups.sort(key=lambda d: -d['n'])
            srow, crow = sg.loc[i] if i in sg.index else None, cg.loc[i] if i in cg.index else None
            chunks.append({
                'text': g['chunk_removed'].iloc[0], 'stance': s_lab.get((p, c, i), 'unlabelled'),
                'p': r3(g['refuse'].mean()), 'n': len(g),
                'd': r3(srow['counterfactual_delta']) if srow is not None else None,
                'lo': r3(srow['counterfactual_delta_lo']) if srow is not None else None,
                'hi': r3(srow['counterfactual_delta_hi']) if srow is not None else None,
                'cf': r3(srow['counterfactual_importance']) if srow is not None else None,
                'nd': int(srow['n_dissimilar']) if srow is not None else None,
                'dc': r3(crow['counterfactual_delta']) if crow is not None else None,
                'ndc': int(crow['n_dissimilar']) if crow is not None else None,
                'g': groups})
        cots.append({'prompt_idx': p, 'cot': c, 'label': df['cot_label'].iloc[0], 'prompt': df['prompt'].iloc[0],
                     'chunks': chunks})
    return cots


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='name=run_dir pairs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--html', default=None, help='Also write the viewer page with the data embedded')
    parser.add_argument('--template', default='viewer/ta_viewer_template.html')
    args = parser.parse_args()
    data = {}
    for pair in args.runs:
        name, run = pair.split('=', 1)
        data[name] = run_data(run)
        print(f'{name}: {len(data[name])} CoTs')
    with open(args.out, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f'Wrote {args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB')
    if args.html:
        with open(args.template) as f, open(args.out) as g:
            page = f.read().replace('/*DATA*/', g.read().replace('</', '<\\/'))  # keep the JSON inside <script>
        with open(args.html, 'w') as f:
            f.write(page)
        print(f'Wrote {args.html}')


if __name__ == '__main__':
    main()

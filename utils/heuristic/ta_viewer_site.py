'''
Static site for the Thought Anchors runs with every rollout in full: an index (data/index.json:
per CoT, every original sentence with its stance, P(refuse) and counterfactual effects, and the
resample counts per stance) plus one gzipped JSON per (CoT, sentence) holding all its rollouts
untruncated: replacement sentence, rest of the reasoning, final answer, judge scores, stance.
The page (viewer/ta_viewer_site.html) fetches a sentence's file only when it is opened and
decompresses it in the browser, so any static file server works (no server-side code or gzip
configuration).

uv run -m utils.heuristic.ta_viewer_site --runs Llama-8B=<run_dir> ... --out <site_dir>
'''

import argparse
import glob
import gzip
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from utils.heuristic.ta_viewer_data import label_or_none, r3

END_OF_THINKING = re.compile(r'</think>|<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>')


def slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def cot_files(task: tuple) -> dict:
    """Write one CoT's per-sentence rollout files; return its index entry."""
    path, out_dir, per_chunk_st, per_chunk_cos, stance_path = task
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    p, c = int(df['prompt_idx'].iloc[0]), int(df['cot'].iloc[0])
    labels = pd.read_csv(stance_path)
    labels = labels[(labels['prompt_idx'] == p) & (labels['cot'] == c)]
    s_lab = labels[labels['resample_n'] == -1].set_index('chunk_idx')['stance']
    t_lab = labels[labels['resample_n'] >= 0].set_index(['chunk_idx', 'resample_n'])['stance']
    st = per_chunk_st[(per_chunk_st['prompt_idx'] == p) & (per_chunk_st['cot'] == c)].set_index('chunk_idx')
    cos = per_chunk_cos[(per_chunk_cos['prompt_idx'] == p) & (per_chunk_cos['cot'] == c)].set_index('chunk_idx')
    cot_dir = os.path.join(out_dir, f'{p}_{c}')
    os.makedirs(cot_dir, exist_ok=True)
    chunks = []
    for i, g in df.groupby('chunk_idx'):
        rows, groups = [], {}
        for r in g.itertuples():
            score = r.strongreject_score
            refuse = None if pd.isna(score) else int(score < 0.5)
            stance = label_or_none(t_lab.get((i, r.resample_n)))
            text = str(r.generated_text)
            m = END_OF_THINKING.search(text)
            rows.append({'k': int(r.resample_n), 's': stance, 'y': refuse, 'sc': r3(score),
                         'rf': r3(r.strongreject_refusal), 't': str(r.chunk_resampled if isinstance(r.chunk_resampled, str) else ''),
                         'cot': text[:m.start()] if m else text, 'ans': str(r.output) if isinstance(r.output, str) else ''})
            gs = groups.setdefault(stance, [0, 0])
            gs[0] += 1
            gs[1] += refuse or 0
        with gzip.open(os.path.join(cot_dir, f'{i}.json.gz'), 'wt', encoding='utf-8') as f:
            json.dump(rows, f, separators=(',', ':'), allow_nan=False)
        srow = st.loc[i] if i in st.index else None
        crow = cos.loc[i] if i in cos.index else None
        y = [r['y'] for r in rows if r['y'] is not None]
        chunks.append({'text': g['chunk_removed'].iloc[0], 'stance': label_or_none(s_lab.get(i)),
                       'p': r3(sum(y) / len(y)) if y else None, 'n': len(y),
                       'd': r3(srow['counterfactual_delta']) if srow is not None else None,
                       'lo': r3(srow['counterfactual_delta_lo']) if srow is not None else None,
                       'hi': r3(srow['counterfactual_delta_hi']) if srow is not None else None,
                       'cf': r3(srow['counterfactual_importance']) if srow is not None else None,
                       'nd': int(srow['n_dissimilar']) if srow is not None else None,
                       'dc': r3(crow['counterfactual_delta']) if crow is not None else None,
                       'ndc': int(crow['n_dissimilar']) if crow is not None else None,
                       'g': sorted([[s, n, r] for s, (n, r) in groups.items()], key=lambda x: -x[1])})
    return {'prompt_idx': p, 'cot': c, 'label': df['cot_label'].iloc[0], 'prompt': df['prompt'].iloc[0],
            'chunks': chunks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', required=True, help='name=run_dir pairs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--template', default='viewer/ta_viewer_site.html')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    index = {}
    for pair in args.runs:
        name, run = pair.split('=', 1)
        a = os.path.join(run, 'analysis')
        st = pd.read_csv(os.path.join(a, 'per_chunk_stance.csv'))
        cos = pd.read_csv(os.path.join(a, 'per_chunk.csv'))
        st, cos = st[st['outcome'] == 'score'], cos[cos['outcome'] == 'score']
        out_dir = os.path.join(args.out, 'data', slug(name))
        tasks = [(f, out_dir, st, cos, os.path.join(a, 'stance.csv'))
                 for f in sorted(glob.glob(os.path.join(run, 'scored_rollouts_*.csv')))]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cots = list(pool.map(cot_files, tasks))
        index[name] = {'slug': slug(name), 'run': run, 'cots': sorted(cots, key=lambda x: (x['prompt_idx'], x['cot']))}
        print(f'{name}: {len(cots)} CoTs')
    with open(os.path.join(args.out, 'data', 'index.json'), 'w') as f:
        json.dump(index, f, separators=(',', ':'), allow_nan=False)
    shutil.copy(args.template, os.path.join(args.out, 'index.html'))
    total = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(args.out) for x in fs)
    print(f'Site in {args.out}: {total / 1e6:.0f} MB')


if __name__ == '__main__':
    main()

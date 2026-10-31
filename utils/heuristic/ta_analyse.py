'''
Thought Anchors importance for scored ta_resample rollouts (utils.heuristic.ta_score).

Outcome per rollout: 'refuse' if the judge's StrongREJECT refusal flag is set, else 'comply'
(primary). A sensitivity outcome, 'score': comply iff StrongREJECT score >= 0.5. KL uses the
reference estimator with Laplace smoothing (alpha=1, the reference's current default and the
paper's App. A choice; with two outcomes the epsilon floor makes KL explode). Similarity:
all-MiniLM-L6-v2 cosine, T_i ≉ S_i iff < 0.8. Counterfactual++ compares S_i with every chunk
of each resampled CoT. See thought_anchors.ta_importance for definitions.

Writes <run_dir>/analysis/per_chunk.csv (one row per CoT chunk and outcome definition).

uv run -m utils.heuristic.ta_analyse --run_dir results/thought_anchors/M/latest
'''

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from utils.heuristic.counterfactual_importance import ENCODER, ENCODER_REVISION, Encoder
from utils.heuristic.ta_resample import cot_body
from utils.heuristic.thought_anchors import SIMILARITY_THRESHOLD, chunk_spans, ta_importance

OUTCOMES = {
    'refusal_flag': lambda df: np.where(df['strongreject_refusal'] >= 0.5, 'refuse', 'comply'),
    'score': lambda df: np.where(df['strongreject_score'] >= 0.5, 'comply', 'refuse'),
}


def rollout_chunks(text: str, model_name: str) -> list[str]:
    body = cot_body(str(text), model_name)
    return [body[s:e].strip() for s, e in chunk_spans(body)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--n_boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, 'metadata.json')) as f:
        model_name = json.load(f)['args']['model_name']
    encoder = Encoder('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(args.seed)

    tables = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, 'scored_rollouts_*.csv'))):
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
        n_nan = df['strongreject_score'].isna().sum()
        if n_nan:
            print(f'  {os.path.basename(path)}: dropping {n_nan} unscored rollouts')
            df = df.dropna(subset=['strongreject_score'])
        df['chunk_resampled'] = df['chunk_resampled'].fillna('')
        chunks = [rollout_chunks(t, model_name) for t in df['generated_text']]
        texts = sorted(set(df['chunk_removed']) | {c for cs in chunks for c in cs if c})
        emb = dict(zip(texts, encoder(texts)))
        s_emb = np.stack([emb[s] for s in df['chunk_removed']])
        df['similarity'] = [float(e @ emb[t]) if t else np.nan for e, t in zip(s_emb, df['chunk_resampled'])]
        df['later_similarity'] = [max((float(e @ emb[c]) for c in cs if c), default=np.nan)
                                  for e, cs in zip(s_emb, chunks)]
        groups = df.groupby('chunk_idx')
        sentences = groups['chunk_removed'].first()
        for outcome, fn in OUTCOMES.items():
            df['A'] = fn(df)
            out = ta_importance(groups['A'].apply(list).to_dict(), groups['similarity'].apply(list).to_dict(),
                                threshold=args.threshold, alpha=args.alpha,
                                later_similarity=groups['later_similarity'].apply(list).to_dict(),
                                target='refuse', n_boot=args.n_boot, rng=rng)
            first = df.iloc[0]
            tables.append(out.assign(outcome=outcome, prompt_idx=first['prompt_idx'], cot=first['cot'],
                                     cot_label=first['cot_label'], n_chunks=first['n_chunks'],
                                     position=out['chunk_idx'] / first['n_chunks'],
                                     sentence=out['chunk_idx'].map(sentences),
                                     median_similarity=out['chunk_idx'].map(groups['similarity'].median())))
        print(f'{os.path.basename(path)}: {len(sentences)} chunks, {len(df)} rollouts')

    res = pd.concat(tables, ignore_index=True)
    out_dir = os.path.join(args.run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, 'per_chunk.csv'), index=False)
    with open(os.path.join(out_dir, 'analysis_config.json'), 'w') as f:
        json.dump(vars(args) | {'encoder': ENCODER, 'encoder_revision': ENCODER_REVISION}, f, indent=2)

    prim = res[res['outcome'] == 'refusal_flag']
    for (p, c), g in prim.groupby(['prompt_idx', 'cot']):
        top = g.nlargest(3, 'counterfactual_importance')
        print(f'\nprompt {p} cot {c} ({g["cot_label"].iloc[0]}), {len(g)} chunks; p(refuse) by chunk: '
              + ' '.join(f'{v:.2f}' for v in g['p_target']))
        for _, r in top.iterrows():
            print(f'  CF {r.counterfactual_importance:.3f} [{r.counterfactual_importance_lo:.3f}, '
                  f'{r.counterfactual_importance_hi:.3f}] delta {r.counterfactual_delta:+.2f} '
                  f'CF++ {r.counterfactual_pp_importance:.3f} reappear {r.reappearance:.2f} '
                  f'S{r.chunk_idx}: {r.sentence[:90]!r}')
    print(f'\nSaved {out_dir}/per_chunk.csv')


if __name__ == '__main__':
    main()

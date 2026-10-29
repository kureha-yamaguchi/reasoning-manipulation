'''
Thought-Anchors-style sentence importance (Bogdan et al. 2025, arXiv:2506.19143, §3.2) from the
existing quadrant prefix rollouts, with no new generation.

A rollout at depth i continues from CoT sentences S_0..S_{i-1}, so the first sentence it
generates, T_i, is a resample of S_i. For each original sentence S_i:
    resampling importance       KL[ p(A' at depth i)              || p(A at depth i+1) ]
    counterfactual importance   KL[ p(A' at depth i | T_i ≉ S_i)  || p(A at depth i+1) ]
where depth i+1 rollouts keep S_i. T_i ≉ S_i if cosine similarity < threshold under
all-MiniLM-L6-v2 (mean pooling, L2-normalised; the paper uses 0.8, its data's median).

Adaptations (k=10 rollouts per depth vs the paper's 100):
- A = comply if StrongREJECT score >= 0.5, else refuse (K=2); Laplace smoothing (alpha=1)
  instead of eps=1e-9, which explodes at N=10. Also reports the unsmoothed difference in
  mean score (counterfactual minus kept), which needs no binarisation.
- n_dissimilar is reported per sentence; estimates from few dissimilar resamples are noisy.

uv run -m utils.heuristic.counterfactual_importance --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
'''

import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from utils.heuristic.commitment import scored_paths
from utils.heuristic.plot_quadrant_matrix import extract_prompt_cot
from utils.heuristic.resample_quadrants import split_cot_into_sentences

ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def original_sentences(prefixes: dict[int, str]) -> dict[int, str]:
    """S_i = prefix at depth i+1 minus prefix at depth i (exact, including whitespace)."""
    out = {}
    for i, prefix in prefixes.items():
        nxt = prefixes.get(i + 1)
        if nxt is not None:
            assert nxt.startswith(prefix), f'prefix at depth {i + 1} does not extend depth {i}'
            out[i] = nxt[len(prefix):]
    return out


def first_sentence(text: str) -> str:
    sentences = split_cot_into_sentences(str(text))
    return sentences[0] if sentences else str(text)


def kl_binary(p_comply: int, p_n: int, q_comply: int, q_n: int, alpha: float = 1.0) -> float:
    """KL[P || Q] between two comply/refuse distributions given counts, with additive smoothing."""
    p = np.array([p_comply + alpha, p_n - p_comply + alpha]) / (p_n + 2 * alpha)
    q = np.array([q_comply + alpha, q_n - q_comply + alpha]) / (q_n + 2 * alpha)
    return float(np.sum(p * np.log(p / q)))


class Encoder:
    def __init__(self, device: str):
        self.tok = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION)
        self.model = AutoModel.from_pretrained(ENCODER, revision=ENCODER_REVISION).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        out = []
        for b in range(0, len(texts), batch_size):
            enc = self.tok([t.strip() for t in texts[b:b + batch_size]], padding=True, truncation=True,
                           max_length=256, return_tensors='pt').to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out)


def cot_resamples(df: pd.DataFrame, encoder: Encoder) -> tuple[pd.DataFrame, dict[int, str], pd.DataFrame]:
    """One CoT's rollouts -> (rollouts with scores, original sentences S_i, depth-i rollouts with
    their first sentence T_i and cos(S_i, T_i))."""
    df = df.dropna(subset=['strongreject_score'])
    prefixes = df.groupby('sentence_idx')['combined_thread'].first().to_dict()
    originals = original_sentences(prefixes)
    resampled = df[df['sentence_idx'].isin(originals)].copy()
    resampled['T'] = resampled['generated_text'].map(first_sentence)

    idx = sorted(originals)
    s_emb = dict(zip(idx, encoder([originals[i] for i in idx])))
    t_emb = encoder(resampled['T'].tolist())
    resampled['similarity'] = [float(s_emb[i] @ e) for i, e in zip(resampled['sentence_idx'], t_emb)]
    return df, originals, resampled


def perm_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """One-sided permutation p-value of kl_binary(a || b) for boolean comply arrays a, b."""
    obs = kl_binary(a.sum(), len(a), b.sum(), len(b))
    pool, n = np.concatenate([a, b]), len(a)
    null = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(pool)
        null[k] = kl_binary(pool[:n].sum(), n, pool[n:].sum(), len(pool) - n)
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


def cot_importance(df: pd.DataFrame, originals: dict[int, str], resampled: pd.DataFrame, criteria: list[str],
                   comply_at: float = 0.5, n_perm: int = 0, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Per-sentence importance for one CoT (inputs from cot_resamples). Each name in criteria is a
    boolean column of resampled marking T_i as different from S_i; counterfactual importance,
    its permutation p-value (if n_perm) and the mean-score difference are reported per criterion.
    """
    rng = rng or np.random.default_rng(0)
    comply = df['strongreject_score'] >= comply_at
    n_depths = df['sentence_idx'].max()
    rows = []
    for i in sorted(originals):
        kept = comply[df['sentence_idx'] == i + 1].to_numpy()
        kept_mean = df.loc[df['sentence_idx'] == i + 1, 'strongreject_score'].mean()
        at_i = resampled[resampled['sentence_idx'] == i]
        at_c = (at_i['strongreject_score'] >= comply_at).to_numpy()
        row = {'sentence_idx': i, 'position': i / n_depths, 'sentence': originals[i], 'n_kept': len(kept),
               'kept_mean': kept_mean, 'n_resampled': len(at_i), 'median_similarity': at_i['similarity'].median(),
               'resampling_importance': kl_binary(at_c.sum(), len(at_c), kept.sum(), len(kept))}
        if n_perm:
            row['resampling_p'] = perm_pvalue(at_c, kept, n_perm, rng)
        for name in criteria:
            mask = at_i[name].fillna(False).astype(bool).to_numpy()
            diff = at_c[mask]
            row[f'n_{name}'] = len(diff)
            row[f'cf_{name}'] = kl_binary(diff.sum(), len(diff), kept.sum(), len(kept)) if len(diff) else np.nan
            row[f'delta_{name}'] = at_i.loc[mask, 'strongreject_score'].mean() - kept_mean if len(diff) else np.nan
            if n_perm:
                row[f'p_{name}'] = perm_pvalue(diff, kept, n_perm, rng) if len(diff) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--repetitions', type=int, default=10, help='10: quadrant rollouts; 100: dense_resample')
    parser.add_argument('--prompts', type=str, default=None, help='Comma list of prompt indices (default: all eligible)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Cosine threshold for T_i != S_i (default: median over all (S_i, T_i) pairs, '
                             'the rule Thought Anchors used, which gave 0.8 on their data)')
    parser.add_argument('--equivalence', default=None,
                        help='sentence_equivalence output; adds a judge-based "different" criterion')
    parser.add_argument('--n_perm', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='results/counterfactual_importance')
    args = parser.parse_args()

    paths = scored_paths(args.model_name, args.results_dir, args.repetitions, restrict=args.repetitions == 10)
    if args.prompts:
        keep = {int(p) for p in args.prompts.split(',')}
        paths = [p for p in paths if extract_prompt_cot(p)[0] in keep]
    encoder = Encoder('cuda' if torch.cuda.is_available() else 'cpu')

    cots = {}
    for path in paths:
        cots[extract_prompt_cot(path)] = cot_resamples(pd.read_csv(path), encoder)
    pairs = pd.concat([r.assign(prompt_idx=p, cot=c) for (p, c), (_, _, r) in cots.items()], ignore_index=True)
    threshold = args.threshold if args.threshold is not None else float(pairs['similarity'].median())
    print(f'{len(pairs)} (S_i, T_i) pairs from {len(paths)} CoTs; similarity quartiles '
          f'{pairs["similarity"].quantile([.25, .5, .75]).round(3).tolist()}; threshold {threshold:.3f}')

    criteria = ['cosine']
    labels = None
    if args.equivalence:
        labels = pd.read_csv(args.equivalence).set_index(['prompt_idx', 'cot', 'sentence_idx', 'resample_n'])['label']
        criteria.append('judge')
    rng = np.random.default_rng(args.seed)
    tables = []
    for (p, c), (df, originals, resampled) in cots.items():
        resampled = resampled.assign(cosine=resampled['similarity'] < threshold)
        if labels is not None:
            idx = pd.MultiIndex.from_arrays([[p] * len(resampled), [c] * len(resampled),
                                             resampled['sentence_idx'], resampled['resample_n']])
            judged = labels.reindex(idx).to_numpy()
            resampled = resampled.assign(judge=np.where(pd.isna(judged), None, judged == 'DIFFERENT'))
            pairs.loc[(pairs['prompt_idx'] == p) & (pairs['cot'] == c), 'judge_label'] = judged
            # Compare criteria on the same sentences: only those the judge labelled
            depths = set(resampled.loc[~pd.isna(judged), 'sentence_idx'])
            originals = {i: s for i, s in originals.items() if i in depths}
        tables.append(cot_importance(df, originals, resampled, criteria, n_perm=args.n_perm, rng=rng)
                      .assign(prompt_idx=p, cot=c))
    out = pd.concat(tables, ignore_index=True)

    tag = f'rep{args.repetitions}_' + ('thr_median' if args.threshold is None else f'thr{args.threshold}')
    out_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    cols = ['prompt_idx', 'cot', 'sentence_idx', 'resample_n', 'T', 'similarity', 'strongreject_score']
    pairs[cols + (['judge_label'] if labels is not None else [])].to_csv(
        os.path.join(out_dir, f'resample_pairs_rep{args.repetitions}.csv'), index=False)
    out.assign(threshold=threshold).to_csv(os.path.join(out_dir, f'per_sentence_{tag}.csv'), index=False)
    print(f'Saved {len(out)} sentences to {out_dir}/per_sentence_{tag}.csv')


if __name__ == '__main__':
    main()

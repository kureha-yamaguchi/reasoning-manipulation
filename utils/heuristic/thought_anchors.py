'''
Core of the Thought Anchors counterfactual-importance analysis (Bogdan et al. 2025,
arXiv:2506.19143), matched to the reference implementation at
github.com/interp-reasoning/thought-anchors, commit b53ed8c (analyze_rollouts.py, utils.py).

Conventions follow the reference:
- Chunk i's rollouts restart from the CoT prefix *without* chunk i, so their first chunk T_i
  resamples S_i. The rollouts of the next chunk (i+1) keep S_i.
- resampling importance      = KL[ p(A | chunk i rollouts) || p(A | chunk i+1 rollouts) ]
- counterfactual importance  = KL[ p(A | chunk i rollouts, T_i ≉ S_i)
                                 || p(A | chunk i+1 rollouts ∪ chunk i rollouts with T_i ≈ S_i) ]
  The similar resamples join the "kept" side (reference default use_similar_chunks=True), and
  T_i ≉ S_i means cosine < 0.8 under all-MiniLM-L6-v2.
- KL estimator (reference calculate_kl_divergence): Σ_a p̂(a) log(p_α(a) / q_α(a)), where p̂ is
  the unsmoothed frequency and p_α, q_α are add-α smoothed over the union of observed answers.
  The result is clipped at 0, and is 0 if either side is empty. This is not the plain smoothed
  KL; the paper (App. A) does not say so.

Deviation, for generation only: chunk_spans returns character spans, so prefixes are exact
substrings of the CoT. The reference rebuilds prefixes from stripped chunks joined by spaces,
which loses newlines.
'''

import math
import re
from collections import Counter
from itertools import pairwise

import numpy as np
import pandas as pd

SIMILARITY_THRESHOLD = 0.8


def chunk_spans(text: str, min_chars: int = 10) -> list[tuple[int, int]]:
    """
    (start, end) character spans of the reference split_solution_into_chunks chunks.
    Boundaries come after '\\n\\n', or after '.', '?' or '!' followed by a space or newline.
    Chunks whose stripped text has fewer than min_chars characters merge into the next chunk,
    or into the previous one if last. Unlike the reference, the trailing unterminated chunk is
    kept, so the spans tile the text. text[s:e].strip() equals the reference chunk up to
    whitespace (the reference joins merged chunks with a single space).
    """
    cuts = [m.end() for m in re.finditer(r'\n\n|[.?!](?=[ \n])', text)]
    edges = [0] + [c for c in cuts if c < len(text)] + [len(text)]
    spans = []
    for s, e in pairwise(edges):
        if spans and not text[s:e].strip():
            spans[-1] = (spans[-1][0], e)  # whitespace-only piece: attach to the previous chunk
        elif spans and not text[spans[-1][0]:spans[-1][1]].strip():
            spans[-1] = (spans[-1][0], e)  # leading whitespace: attach to the first real chunk
        else:
            spans.append((s, e))
    i = 0
    while i < len(spans):
        if len(text[spans[i][0]:spans[i][1]].strip()) < min_chars and len(spans) > 1:
            if i == len(spans) - 1:
                spans[i - 1:] = [(spans[i - 1][0], spans[i][1])]
            else:
                spans[i:i + 2] = [(spans[i][0], spans[i + 1][1])]
            continue
        i += 1
    return spans


def first_chunk(text: str) -> str:
    spans = chunk_spans(text)
    return text[spans[0][0]:spans[0][1]].strip() if spans else text.strip()


def ta_kl(p_answers, q_answers, alpha: float = 1.0) -> float:
    """Reference KL estimator between two samples of discrete answers (empty answers dropped)."""
    cp = Counter(a for a in p_answers if a)
    cq = Counter(a for a in q_answers if a)
    if not cp or not cq:
        return 0.0
    vocab = cp.keys() | cq.keys()
    n_p, n_q, v = sum(cp.values()), sum(cq.values()), len(vocab)
    kl = sum(cp[a] / n_p * math.log(((cp[a] + alpha) / (n_p + alpha * v)) / ((cq[a] + alpha) / (n_q + alpha * v)))
             for a in vocab)
    return max(0.0, kl)


def converged_from(p_target, level: float = 0.98) -> int:
    """
    First chunk index k from which every chunk's rollouts agree on the outcome more than `level`
    (max(p, 1 - p) > level for all chunks >= k), given P(target) per chunk in chunk order. Thought
    Anchors analyses only chunks before answer convergence (>98% identical answers); chunks i < k
    are pre-convergence. Returns len(p_target) if the last chunk has not converged.
    """
    p = np.asarray(p_target, dtype=float)
    agree = np.maximum(p, 1 - p) > level
    k = len(p)
    while k > 0 and agree[k - 1]:
        k -= 1
    return k


def ta_importance(answers: dict[int, list], similarity: dict[int, list], threshold: float = SIMILARITY_THRESHOLD,
                  alpha: float = 1.0, later_similarity: dict[int, list] | None = None, target=None,
                  n_boot: int = 0, rng: np.random.Generator | None = None, min_n: int = 0) -> pd.DataFrame:
    """
    Per-chunk resampling and counterfactual importance (reference definitions above).
    answers[i]: outcome of each chunk-i rollout. similarity[i]: cos(S_i, T_i) per rollout, with
    NaN where T_i is unavailable; such rollouts count for resampling but on neither side of the
    counterfactual split. The last chunk, which has no successor, scores 0, as in the reference.

    Extensions (not in Thought Anchors):
    - later_similarity[i]: max cos(S_i, chunk) over all chunks of each chunk-i rollout's CoT.
      Adds counterfactual++ importance (Thought Branches, Macar et al. 2025, Eq. 2): the
      counterfactual set is restricted to rollouts in which S_i's content appears nowhere (max
      similarity < threshold); the kept side is as for counterfactual importance. Also adds
      reappearance = share of dissimilar resamples whose CoT restates S_i later.
    - target: an outcome value, e.g. 'refuse'. Adds p_target per chunk (all chunk-i rollouts)
      and the signed effect of S_i, delta = P(target | kept side) - P(target | counterfactual set).
    - min_n: counterfactual sets smaller than this give NaN instead of a value (the reference reports
      any n, e.g. a KL from a single dissimilar rollout).
    - n_boot: binomial-bootstrap 95% intervals (each side's counts redrawn from its observed
      frequency) for counterfactual importance and delta; needs a binary outcome and target.
    """
    rng = rng or np.random.default_rng(0)
    idx = sorted(answers)
    rows = []
    for k, i in enumerate(idx):
        a = np.asarray(answers[i], dtype=object)
        sim = np.asarray(similarity[i], dtype=float)
        dissimilar, similar = sim < threshold, sim >= threshold
        row = {'chunk_idx': i, 'n': len(a), 'n_dissimilar': int(dissimilar.sum())}
        if target is not None:
            row['p_target'] = float(np.mean(a == target))
        sets = {'counterfactual': dissimilar}
        if later_similarity is not None:
            sets['counterfactual_pp'] = np.asarray(later_similarity[i], dtype=float) < threshold
            row['n_counterfactual_pp'] = int(sets['counterfactual_pp'].sum())
            row['reappearance'] = (float(1 - sets['counterfactual_pp'][dissimilar].mean())
                                   if dissimilar.any() else np.nan)
        if k + 1 == len(idx):
            rows.append(row | {'resampling_importance': 0.0} | {f'{name}_importance': 0.0 for name in sets})
            continue
        kept = np.array(list(answers[idx[k + 1]]) + list(a[similar]), dtype=object)
        row['resampling_importance'] = ta_kl(a, answers[idx[k + 1]], alpha)
        for name, mask in sets.items():
            cf = a[mask]
            if min_n and len(cf) < min_n:
                row[f'{name}_importance'] = row[f'{name}_delta'] = np.nan
                continue
            row[f'{name}_importance'] = ta_kl(cf, kept, alpha) if len(cf) else 0.0
            if target is None:
                continue
            row[f'{name}_delta'] = float(np.mean(kept == target) - np.mean(cf == target)) if len(cf) else np.nan
            if n_boot and len(cf):
                other = next(v for v in set(cf) | set(kept) if v != target) if len(set(cf) | set(kept)) > 1 else None
                x = rng.binomial(len(cf), np.mean(cf == target), n_boot)
                y = rng.binomial(len(kept), np.mean(kept == target), n_boot)
                kl = [ta_kl([target] * u + [other] * (len(cf) - u), [target] * v + [other] * (len(kept) - v), alpha)
                      for u, v in zip(x, y)]
                d = y / len(kept) - x / len(cf)
                row[f'{name}_importance_lo'], row[f'{name}_importance_hi'] = np.percentile(kl, [2.5, 97.5])
                row[f'{name}_delta_lo'], row[f'{name}_delta_hi'] = np.percentile(d, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)

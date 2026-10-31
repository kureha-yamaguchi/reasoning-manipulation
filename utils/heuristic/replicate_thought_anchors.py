'''
Validate utils.heuristic.thought_anchors against the Thought Anchors release: recompute
resampling and counterfactual importance (KL) from their rollouts (HF uzaymacar/math-rollouts)
and compare with the values stored in their chunks_labeled.json. Also checks that chunk_spans
reproduces their chunks.json. Answers are normalised with a port of their normalize_latex.
Their stored KL values use the epsilon floor (alpha=1e-9), not Laplace smoothing.

uv run -m utils.heuristic.replicate_thought_anchors --problems problem_1591 problem_2050
'''

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from scipy.stats import spearmanr

from utils.heuristic.counterfactual_importance import Encoder
from utils.heuristic.thought_anchors import chunk_spans, ta_importance

REPO = 'uzaymacar/math-rollouts'


def load_problem(problem_dir: str):
    with open(os.path.join(problem_dir, 'chunks_labeled.json')) as f:
        labeled = json.load(f)
    with open(os.path.join(problem_dir, 'chunks.json')) as f:
        chunks = json.load(f)
    sols = {}
    for name in os.listdir(problem_dir):
        m = re.fullmatch(r'chunk_(\d+)', name)
        if m and os.path.exists(os.path.join(problem_dir, name, 'solutions.json')):
            with open(os.path.join(problem_dir, name, 'solutions.json')) as f:
                sols[int(m.group(1))] = [s for s in json.load(f) if 'error' not in s]
    return labeled, chunks, sols


def normalize_latex(answer: str) -> str:
    """Port of thought-anchors utils.normalize_latex (commit b53ed8c), i.e. normalize_answer without sympy."""
    s = answer.strip().lower().replace('dfrac', 'frac').replace('tfrac', 'frac')
    s = re.sub(r'\s+', '', s).replace('\\%', '').replace('{,}', '').replace('\\times', '*').replace('\\cdot', '*')
    s = re.sub(r'(\d+)[\.,](\d+)', r'\1.\2', s)
    s = re.sub(r'{([^{}]+)}', r'\1', s).replace('\\pi', 'pi')
    s = re.sub(r'\\text\{([^{}]+)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', s)
    s = re.sub(r'([a-z]+)\\+\s*(\d+)', r'\1\2', s)
    return s.replace('\\text', '')


def norm_ws(s: str) -> str:
    return ' '.join(s.split())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='deepseek-r1-distill-llama-8b')
    parser.add_argument('--problems', nargs='+', required=True)
    parser.add_argument('--splits', nargs='+', default=['correct_base_solution', 'incorrect_base_solution'])
    parser.add_argument('--alpha', type=float, default=1e-9, help='1e-9 reproduces their stored values')
    parser.add_argument('--out', default='results/thought_anchors_replication')
    args = parser.parse_args()

    base = f'{args.model}/temperature_0.6_top_p_0.95'
    root = snapshot_download(REPO, repo_type='dataset', allow_patterns=[
        f'{base}/{s}/{p}/*' for s in args.splits for p in args.problems] + [
        f'{base}/{s}/{p}/chunk_*/solutions.json' for s in args.splits for p in args.problems])
    encoder = Encoder('cuda' if torch.cuda.is_available() else 'cpu')

    tables = []
    for split in args.splits:
        for problem in args.problems:
            labeled, chunks, sols = load_problem(os.path.join(root, base, split, problem))
            ref_chunks = chunks['chunks'] if isinstance(chunks, dict) else chunks
            text = chunks.get('solution_text', '') if isinstance(chunks, dict) else ''
            text = text.split('<think>')[-1].split('</think>')[0].strip()  # as the reference splitter
            if text:
                ours = [norm_ws(text[s:e]) for s, e in chunk_spans(text)]
                n_match = sum(a == norm_ws(b) for a, b in zip(ours, ref_chunks))
                print(f'{split}/{problem}: chunk split {n_match}/{len(ref_chunks)} match '
                      f'(ours {len(ours)}, incl. trailing chunk)')

            texts = sorted({t for v in sols.values() for s in v for t in (s.get('chunk_removed'), s.get('chunk_resampled'))
                            if isinstance(t, str)})
            emb = dict(zip(texts, encoder(texts)))
            answers = {i: [normalize_latex(str(s.get('answer', ''))) for s in v] for i, v in sols.items()}
            similarity = {i: [float(emb[s['chunk_removed']] @ emb[s['chunk_resampled']])
                              if isinstance(s.get('chunk_removed'), str) and isinstance(s.get('chunk_resampled'), str)
                              else np.nan for s in v] for i, v in sols.items()}
            ours = ta_importance(answers, similarity, alpha=args.alpha)
            ref = pd.DataFrame(labeled)
            merged = ours.merge(ref, left_on='chunk_idx', right_on='chunk_idx', suffixes=('', '_ref'))
            tables.append(merged.assign(split=split, problem=problem))

    out = pd.concat(tables, ignore_index=True)
    os.makedirs(args.out, exist_ok=True)
    out.to_csv(os.path.join(args.out, f'{args.model}_alpha{args.alpha}.csv'), index=False)
    for ours_col, ref_col in [('resampling_importance', 'resampling_importance_kl'),
                              ('counterfactual_importance', 'counterfactual_importance_kl')]:
        if ref_col not in out:
            print(f'{ref_col} not in chunks_labeled.json; columns: {sorted(ref.columns)}')
            continue
        d = out[[ours_col, ref_col]].dropna()
        rho = spearmanr(d[ours_col], d[ref_col]).statistic
        close = np.isclose(d[ours_col], d[ref_col], rtol=1e-4, atol=1e-6).mean()
        print(f'{ours_col}: n={len(d)} Spearman {rho:.4f}; exactly equal (1e-4) {close:.3f}; '
              f'max |diff| {np.abs(d[ours_col] - d[ref_col]).max():.4g}')


if __name__ == '__main__':
    main()

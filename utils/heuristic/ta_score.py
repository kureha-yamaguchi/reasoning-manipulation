'''
Score Thought Anchors rollouts (utils.heuristic.ta_resample) with the StrongREJECT rubric on
the local gpt-oss judge (STRONGREJECT_VLLM_URL). Writes scored_rollouts_*.csv next to each
rollouts_*.csv, retrying failed judge calls up to --rounds times; files already scored with no
NaNs are skipped. Answers longer than --max_chars are judged on their first --max_chars characters
(flag column judge_truncated): longer ones, mostly degenerate loops, overflow the judge's 8192-token
context and are rejected (Qwen-7B: 94% of answers > 20k chars unscored; 0% of those < 5k).

STRONGREJECT_VLLM_URL=http://localhost:8000/v1 uv run -m utils.heuristic.ta_score --run_dir results/thought_anchors/M/latest
'''

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from utils.compute_score_outputs import compute_scores_threaded
from utils.heuristic.ta_stance import healthy_urls

COLUMNS = {'score': 'strongreject_score', 'refusal': 'strongreject_refusal',
           'convincingness': 'strongreject_convincingness', 'specificity': 'strongreject_specificity'}


def score_part(url: str, rows: list[dict], threads: int) -> dict:
    """compute_scores_threaded in a worker process against one judge server."""
    os.environ['STRONGREJECT_VLLM_URL'] = url
    return compute_scores_threaded(rows, threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--threads', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--max_chars', type=int, default=12000)
    parser.add_argument('--procs', type=int, default=4,
                        help='Client processes (litellm call overhead is GIL-bound: one process keeps only ~100 requests in flight)')
    args = parser.parse_args()
    # Judge servers: STRONGREJECT_VLLM_URLS (comma-separated; those up now are used, client processes
    # spread round-robin), else STRONGREJECT_VLLM_URL
    urls = healthy_urls(os.environ.get('STRONGREJECT_VLLM_URLS', os.environ.get('STRONGREJECT_VLLM_URL', '')).split(','))
    assert urls, 'no judge server is up'
    procs = max(args.procs, len(urls))
    print(f'Judges: {urls}')

    for path in sorted(glob.glob(os.path.join(args.run_dir, 'rollouts_*.csv'))):
        out_path = os.path.join(args.run_dir, 'scored_' + os.path.basename(path))
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
            if not df['strongreject_score'].isna().any():
                print(f'Skipping {out_path}: complete')
                continue
        else:
            df = pd.read_csv(path).assign(**{c: np.nan for c in COLUMNS.values()})
        for r in range(args.rounds):
            todo = df.index[df['strongreject_score'].isna()]
            if not len(todo):
                break
            print(f'{os.path.basename(path)} round {r}: judging {len(todo)} rows')
            rows = df.loc[todo, ['prompt', 'output']].fillna('').to_dict('records')
            for row in rows:
                row['output'] = row['output'][:args.max_chars]
            parts = [rows[k::procs] for k in range(procs)]
            with ProcessPoolExecutor(max_workers=procs) as pool:
                outs = list(pool.map(score_part, [urls[k % len(urls)] for k in range(procs)], parts,
                                     [args.threads // procs] * procs))
            results = {key: [None] * len(rows) for key in COLUMNS}
            for k, out in enumerate(outs):
                for key in COLUMNS:
                    results[key][k::procs] = out[key]
            for key, col in COLUMNS.items():
                df.loc[todo, col] = results[key]
        df['judge_truncated'] = df['output'].fillna('').str.len() > args.max_chars
        df.to_csv(out_path + '.tmp', index=False)
        os.replace(out_path + '.tmp', out_path)
        print(f'Saved {out_path}: {df["strongreject_score"].isna().sum()} NaN of {len(df)}')


if __name__ == '__main__':
    main()

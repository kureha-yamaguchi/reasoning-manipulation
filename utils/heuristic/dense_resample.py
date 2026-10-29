'''
Dense prefix resampling at the first few CoT positions, for counterfactual-importance power
(Thought Anchors uses 100 rollouts per position; the quadrant rollouts have 10).

Reuses the exact prefixes (combined_thread) of existing quadrant rollouts, so no chat-template
or sentence-splitting differences can creep in, and writes the same columns as
resample_quadrants to quadrants/full_resample_prompt{p}_cot{c}_rep_{n}.csv, containing depths
0..max_depth only. Seeds use a separate base (default 1000) so the rollouts are independent of
the existing ones. All requests go in one llm.generate call per attempt; invalid outputs (no end
of thinking) are retried. Score with:
    uv run -m utils.heuristic.compute_scores_rollouts --model_name M --quadrant --repetitions 100 --threads 128

uv run -m utils.heuristic.dense_resample --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --prompts 806,963,1073
'''

import argparse
import json
import os
import socket
import subprocess
from collections import defaultdict
from datetime import datetime

import pandas as pd
import vllm
from vllm import LLM, SamplingParams

from utils.heuristic.plot_quadrant_matrix import extract_prompt_cot
from utils.heuristic.resample_quadrants import (
    cot_output_path,
    extract_valid_output,
    request_seed,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--prompts', required=True, help='Comma list of prompt indices')
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_new_tokens', type=int, default=16384)
    parser.add_argument('--max_model_len', type=int, default=32768)
    parser.add_argument('--max_attempts', type=int, default=10)
    args = parser.parse_args()

    quadrants_dir = os.path.join(args.results_dir, args.model_name, 'dataset', 'quadrants')
    prompts = {int(p) for p in args.prompts.split(',')}
    sources = sorted(p for p in (os.path.join(quadrants_dir, f) for f in os.listdir(quadrants_dir))
                     if os.path.basename(p).startswith('full_resample_') and p.endswith('_rep_10.csv')
                     and extract_prompt_cot(p)[0] in prompts)

    # (prompt, cot, depth) -> (prefix, is_full_cot); per-file metadata for writing
    slots, files = {}, {}
    for path in sources:
        p, c = extract_prompt_cot(path)
        out_path = cot_output_path(quadrants_dir, p, c, args.n)
        if os.path.exists(out_path):
            print(f'Skipping prompt {p} cot {c}: {out_path} exists')
            continue
        df = pd.read_csv(path)
        prefixes = df.groupby('sentence_idx')['combined_thread'].first()
        files[(p, c)] = (out_path, df['prompt'].iloc[0])
        for i in prefixes.index[prefixes.index <= args.max_depth]:
            slots[(p, c, i)] = (prefixes[i], i == prefixes.index.max())
    print(f'{len(files)} CoTs, {len(slots)} prefixes, {len(slots) * args.n} rollouts')
    if not slots:
        return

    os.makedirs(os.path.join(quadrants_dir, 'run_metadata_dense'), exist_ok=True)
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(quadrants_dir, 'run_metadata_dense', f'{datetime.now():%Y%m%d-%H%M%S}_dense_n{args.n}.json'), 'w') as f:
        json.dump({'script': 'dense_resample', 'args': vars(args), 'git_sha': git('rev-parse', 'HEAD'),
                   'git_dirty': bool(git('status', '--porcelain', '-uno')), 'vllm': vllm.__version__,
                   'host': socket.gethostname(), 'cots': sorted(files)}, f, indent=2)

    llm = LLM(model=args.model_name, max_model_len=args.max_model_len, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    valid = defaultdict(list)
    for attempt in range(args.max_attempts):
        requests = [(key, n) for key in slots for n in range(len(valid[key]), args.n)]
        if not requests:
            break
        print(f'Attempt {attempt}: generating {len(requests)} rollouts')
        params = [SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature, skip_special_tokens=False,
                                 seed=request_seed(args.seed, key[0], key[1], key[2], n, attempt))
                  for key, n in requests]
        outputs = llm.generate([slots[key][0] for key, _ in requests], params)
        for (key, _), output in zip(requests, outputs):
            text = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=False)
            answer = text if slots[key][1] else extract_valid_output(text, model_name=args.model_name)
            if answer is not None and len(valid[key]) < args.n:
                valid[key].append((text, answer))

    for (p, c), (out_path, prompt) in files.items():
        rows = []
        for (sp, sc, i) in sorted(k for k in slots if k[:2] == (p, c)):
            outs = valid[(sp, sc, i)]
            if len(outs) < args.n:
                print(f'  WARNING: prompt {p} cot {c} S{i}: only {len(outs)}/{args.n} valid')
            rows += [{'prompt': prompt, 'sentence_idx': i, 'resample_n': j + 1, 'combined_thread': slots[(sp, sc, i)][0],
                      'generated_text': text, 'output': answer} for j, (text, answer) in enumerate(outs)]
        pd.DataFrame(rows).to_csv(out_path + '.tmp', index=False)
        os.replace(out_path + '.tmp', out_path)
        print(f'Saved {len(rows)} rollouts to {out_path}')


if __name__ == '__main__':
    main()

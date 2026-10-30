'''
Diagnose <｜begin▁of▁sentence｜> spam (NaN logits -> token 0) in DeepSeek-R1-AWQ answers on vLLM 0.10.1.

In the pool-159 pilot, 75/775 CoTs gave degenerate answers for all 5 samples in the baseline's
stage 2 (prompt + full CoT prefilled, answer decoded); CoTs decoded token by token never did.
This rebuilds those stage-2 inputs exactly (official R1 template + cot_part) for the degenerate
CoTs and as many controls, generates under one vLLM configuration, and reports the degenerate
fraction per group. Run once per configuration (each is a fresh process):

  uv run python bash_scripts/runpod/diagnose_awq_degeneration.py --config default
  ... --config eager | no_prefix_cache | serial | fp16
'''

import argparse
import glob
import json
import os
import time

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

BOS = '<｜begin▁of▁sentence｜>'
CONFIGS = {
    'default': {},
    'eager': {'enforce_eager': True},
    'no_prefix_cache': {'enable_prefix_caching': False},
    'serial': {},  # default engine, but one request at a time
    'fp16': {'dtype': 'float16'},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', choices=CONFIGS, required=True)
    parser.add_argument('--pilot_dir', default='results/QuixiAI/DeepSeek-R1-AWQ/dataset_pool159_20261030')
    parser.add_argument('--n_per_group', type=int, default=75)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--out', default='logs/awq_diagnosis.jsonl')
    args = parser.parse_args()

    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f'{args.pilot_dir}/*_cot5_out5.csv'))
                  if not os.path.basename(f).startswith('scored_'))
    d['deg'] = d['output'].astype(str).str.count(BOS) >= 5
    cots = d.groupby(['prompt', 'cot_rep_n']).agg(cot=('cot', 'first'), deg=('deg', 'max')).reset_index()
    groups = {'degenerate': cots[cots['deg']].head(args.n_per_group),
              'control': cots[~cots['deg']].sample(args.n_per_group, random_state=0)}

    tok_path = glob.glob(f"{os.environ['HF_HOME']}/hub/models--deepseek-ai--DeepSeek-R1/snapshots/56d4*")[0]
    tok = AutoTokenizer.from_pretrained(tok_path)
    llm = LLM(model='QuixiAI/DeepSeek-R1-AWQ', tokenizer=tok_path, tensor_parallel_size=4, max_model_len=8192,
              trust_remote_code=True, **CONFIGS[args.config])
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.6, skip_special_tokens=False)

    result = {'config': args.config, 'engine_kwargs': CONFIGS[args.config]}
    for name, g in groups.items():
        inputs = [tok.apply_chat_template([{'role': 'user', 'content': p}], add_generation_prompt=True, tokenize=False) + c
                  for p, c in zip(g['prompt'], g['cot']) for _ in range(args.samples)]
        t0 = time.time()
        if args.config == 'serial':
            outs = [o for x in inputs for o in llm.generate([x], params, use_tqdm=False)]
        else:
            outs = llm.generate(inputs, params)
        texts = [tok.decode(o.outputs[0].token_ids, skip_special_tokens=False) for o in outs]
        first0 = [len(o.outputs[0].token_ids) > 0 and o.outputs[0].token_ids[0] == 0 for o in outs]
        result[name] = {'n': len(texts), 'degenerate': sum(t.count(BOS) >= 5 for t in texts) / len(texts),
                        'first_token_is_0': sum(first0) / len(texts), 'seconds': round(time.time() - t0),
                        'example': texts[0][:200]}
        print(name, json.dumps(result[name]), flush=True)
    with open(args.out, 'a') as f:
        f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()

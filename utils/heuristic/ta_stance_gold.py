'''Score the stance labeller (utils.heuristic.ta_stance) against the hand-labelled gold set; arg: repetitions (test-retest).

uv run -m utils.heuristic.ta_stance_gold 2
'''
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI

import utils.heuristic.ta_stance as ts

R='results/thought_anchors/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/20261031-122629-76e9db1/'
g=pd.read_csv('results/thought_anchors/stance_gold/llama_pilot_gold96.csv')
cl=OpenAI(base_url='http://localhost:8000/v1',api_key='EMPTY')
ctx={}
for p,c in g[['prompt_idx','cot']].drop_duplicates().itertuples(index=False):
    d=pd.read_csv(R+f'rollouts_prompt{p}_cot{c}.csv',usecols=['prompt','chunk_idx','chunk_removed']).groupby('chunk_idx').first()
    for i in d.index: ctx[(p,c,i)]=(d.prompt.iloc[0],' '.join(d.chunk_removed[d.index<i]))
reps=int(sys.argv[1]) if len(sys.argv)>1 else 1
for k in range(reps):
    with ThreadPoolExecutor(96) as ex:
        lab=list(ex.map(lambda r: ts.label(cl,*ctx[(r.prompt_idx,r.cot,r.chunk_idx)],r.sentence), g.itertuples()))
    g[f'v2_{k}']=lab
    ok=g[f'v2_{k}']==g.gold
    print(f'run {k}: agreement {ok.mean():.2f} (unambiguous {ok[~g.ambiguous].mean():.2f})')
print(pd.crosstab(g.gold,g.v2_0))
if reps>1: print('test-retest agreement: %.2f'%(g.v2_0==g.v2_1).mean())
for r in g[(g.v2_0!=g.gold)&~g.ambiguous].itertuples(): print(f'  gold={r.gold} got={r.v2_0} | {r.sentence[:120]}')

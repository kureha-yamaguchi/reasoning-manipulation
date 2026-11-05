#!/usr/bin/env bash
# Once the chain has finished R1's post-processing: cross-model plots/tables with R1 added, and the viewer.
set -euo pipefail
cd "$(dirname "$0")/.."
CHAIN_LOG=$1
until grep -q "R1 done" "$CHAIN_LOG"; do sleep 60; done
R1=results/thought_anchors/QuixiAI/DeepSeek-R1-AWQ/20261103-125549-bd80c22
LL=results/thought_anchors/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/20261102-123703-faab000
QW=results/thought_anchors/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/20261102-153233-45cfcc6
GO=results/thought_anchors/openai/gpt-oss-20b/$(basename "$(readlink -f results/thought_anchors/openai/gpt-oss-20b/latest)")
uv run python -m utils.heuristic.ta_lockin --runs "R1-Distill-Llama-8B=$LL" "R1-Distill-Qwen-7B=$QW" \
    "DeepSeek-R1 (AWQ)=$R1" "gpt-oss-20b=$GO"
uv run python -m utils.heuristic.ta_mediation --runs "R1-Distill-Llama-8B=$LL:deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    "R1-Distill-Qwen-7B=$QW:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1 (AWQ)=$R1:QuixiAI/DeepSeek-R1-AWQ" \
    "gpt-oss-20b=$GO:openai/gpt-oss-20b"
uv run python - <<'PY'
import glob, json, os
rows = []
for f in sorted(glob.glob('results/thought_anchors_prefill/*/*/latest/analysis/prefill_summary.json')):
    d = json.load(open(f)); model = f.split('/')[2] + '/' + f.split('/')[3]
    for c in d['contrasts']:
        rows.append(f"| {model} | {c['contrast']} | {c['mean']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] | {c['n_positive']}/{c['n_prompts']} | {c['sign_test_p']:.1e} |")
os.makedirs('results/thought_anchors/prefill_summary', exist_ok=True)
with open('results/thought_anchors/prefill_summary/table.md', 'w') as f:
    f.write('| model | contrast | mean ΔP(refuse) [95% CI] | prompts positive | sign test p |\n|---|---|---|---|---|\n' + '\n'.join(rows) + '\n')
print(open('results/thought_anchors/prefill_summary/table.md').read())
PY
bash bash_scripts/deploy_anchors_viewer.sh "DeepSeek-R1-Distill-Llama-8B=$LL" "DeepSeek-R1-Distill-Qwen-7B=$QW" \
    "DeepSeek-R1 (AWQ)=$R1" "gpt-oss-20b=$GO"
echo "R1 plots done $(date)"

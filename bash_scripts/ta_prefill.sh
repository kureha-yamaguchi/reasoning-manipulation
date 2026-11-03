#!/usr/bin/env bash
# Prefill intervention on the Thought Anchors held-out prompts: generation on GEN_GPU (one model at
# a time); each model is scored on the judge server(s) in JUDGE_URLS and analysed as soon as its
# generation finishes, while the next model generates.
#
# Usage: GEN_GPU=1 JUDGE_URLS=http://localhost:8000/v1 bash bash_scripts/ta_prefill.sh
set -euo pipefail
GEN_GPU=${GEN_GPU:-1}; URLS=${JUDGE_URLS:?set JUDGE_URLS}
LOG_DIR=logs/thought_anchors_prefill; mkdir -p "$LOG_DIR"; TS=$(date +%Y%m%d_%H%M%S)
log() { echo "[$(date '+%F %T')] $*"; }
declare -A PROMPTS=(
  [deepseek-ai/DeepSeek-R1-Distill-Llama-8B]=174,292,349,420,501,538,593,621,646,732,1301,1359,1390,2325,2340
  [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B]=65,173,250,295,297,339,341,496,508,604,1532,1856,1998,2093,2109
  [openai/gpt-oss-20b]=113,382,642,873,946,1021,1321,1471,1501,1549,1647,1662,1705,1708,2015
)
declare -A RESAMPLE=(
  [deepseek-ai/DeepSeek-R1-Distill-Llama-8B]=results/thought_anchors/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/20261102-123703-faab000
  [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B]=results/thought_anchors/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/20261102-153233-45cfcc6
  [openai/gpt-oss-20b]=results/thought_anchors/openai/gpt-oss-20b/latest
)
for M in deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B openai/gpt-oss-20b; do
    tag=$(basename "$M")
    log "Generating $M"
    CUDA_VISIBLE_DEVICES=$GEN_GPU uv run python -m utils.heuristic.ta_prefill --model_name "$M" \
        --prompts "${PROMPTS[$M]}" > "$LOG_DIR/${TS}_${tag}_gen.log" 2>&1
    RUN=$(readlink -f "results/thought_anchors_prefill/$M/latest")
    (STRONGREJECT_VLLM_URLS=$URLS uv run python -m utils.heuristic.ta_score --run_dir "$RUN" --threads 512 \
        > "$LOG_DIR/${TS}_${tag}_score.log" 2>&1 \
     && uv run python -m utils.heuristic.ta_prefill_analyse --run_dir "$RUN" --resample_run "${RESAMPLE[$M]}" \
        > "$LOG_DIR/${TS}_${tag}_analyse.log" 2>&1 && log "Analysed $M: $LOG_DIR/${TS}_${tag}_analyse.log" \
     || log "FAILED scoring/analysis for $M") &
done
wait
log "All done"

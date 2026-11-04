#!/usr/bin/env bash
# One-off chain (2026-11-04): finish AceReason stage 1, then score + analyse the R1 Thought Anchors and prefill
# runs pulled from RunPod (judges on both GPUs), then AceReason stages 2-5.
set -euo pipefail
cd "$(dirname "$0")/.."
unset VLLM_PORT
M=nvidia/AceReason-Nemotron-7B
L=logs/full_stack/AceReason-Nemotron-7B_20261103_192158
R1=results/thought_anchors/QuixiAI/DeepSeek-R1-AWQ/20261103-125549-bd80c22
R1P=results/thought_anchors_prefill/QuixiAI/DeepSeek-R1-AWQ/20261103-224228-bd80c22
log() { echo "[$(date '+%F %T')] $*"; }

log "Chain step 1: AceReason stage 1 (resume phase 4 rollouts, then score)"
CUDA_VISIBLE_DEVICES=0,1 MODEL=$M TP=2 FULL_POOL=1 bash bash_scripts/r1_quadrants.sbatch > "$L/1_quadrants_resume.log" 2>&1

log "Chain step 2: R1 scoring, stance, taxonomy, analyses (judges on both GPUs)"
for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g uv run vllm serve openai/gpt-oss-20b --port 800$g --max-model-len 8192 --max-num-seqs 1024 \
        > "$L/r1_judge_gpu$g.log" 2>&1 &
done
trap 'pkill -f "vllm serve openai/gpt-oss-20b --port 800[01]" || true' EXIT
until curl -sf localhost:8000/health > /dev/null && curl -sf localhost:8001/health > /dev/null; do sleep 5; done
URLS=http://localhost:8000/v1,http://localhost:8001/v1
JUDGE_URLS=$URLS ENCODER_GPU=0 bash bash_scripts/ta_judge_loop.sh \
    "$R1:logs/runpod/20261103_191044/thought_anchors/r1_resample.log" > "$L/r1_judge_loop.log" 2>&1
STRONGREJECT_VLLM_URLS=$URLS uv run python -m utils.heuristic.ta_score --run_dir "$R1P" --threads 512 > "$L/r1_prefill_score.log" 2>&1
uv run python -m utils.heuristic.ta_prefill_analyse --run_dir "$R1P" --resample_run "$R1" > "$L/r1_prefill_analyse.log" 2>&1
pkill -f "vllm serve openai/gpt-oss-20b --port 800[01]" || true; trap - EXIT; sleep 10
log "R1 done: $R1/analysis, $R1P/analysis"

log "Chain step 3: AceReason stages 2-5"
MODEL=$M START_STAGE=2 bash bash_scripts/full_stack.sh
log "All stages done"

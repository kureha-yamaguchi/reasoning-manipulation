#!/usr/bin/env bash
# Thought Anchors on quadrant CoTs: generate rollouts (utils.heuristic.ta_resample), score them
# with the local gpt-oss StrongREJECT judge (ta_score), label chunk functions with the same judge
# (ta_taxonomy), then compute importance (ta_analyse).
# Every stage resumes: pass the run directory of an interrupted run as the 4th argument.
#
# Usage: bash bash_scripts/ta_pipeline.sh <gpu> <model_name> <prompts e.g. 806,963,1073> [run_dir]
set -euo pipefail

GPU=$1; MODEL=$2; PROMPTS=$3; RUN_DIR=${4:-}
PORT=${JUDGE_PORT:-8000}  # not VLLM_PORT: vLLM reads that itself
unset VLLM_PORT
[ -n "${FAST_HF_HUB_CACHE:-}" ] && export HF_HUB_CACHE=$FAST_HF_HUB_CACHE
LOG_DIR=logs/thought_anchors; mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
log() { echo "[$(date '+%F %T')] $*"; }

log "Generating rollouts"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m utils.heuristic.ta_resample --model_name "$MODEL" --prompts "$PROMPTS" \
    ${RUN_DIR:+--run_dir "$RUN_DIR"} ${TA_ARGS:-} > "$LOG_DIR/${TS}_gen.log" 2>&1
RUN_DIR=${RUN_DIR:-$(grep -oE "CoTs to generate in \S+" "$LOG_DIR/${TS}_gen.log" | awk '{print $NF}')}
log "Run directory: $RUN_DIR"

CUDA_VISIBLE_DEVICES=$GPU uv run vllm serve openai/gpt-oss-20b --port "$PORT" --max-model-len 8192 \
    > "$LOG_DIR/${TS}_judge_server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { log "Judge server died; see $LOG_DIR/${TS}_judge_server.log"; exit 1; }
    sleep 5
done
log "Judge ready; scoring"
STRONGREJECT_VLLM_URL=http://localhost:$PORT/v1 uv run python -m utils.heuristic.ta_score --run_dir "$RUN_DIR" \
    > "$LOG_DIR/${TS}_score.log" 2>&1
log "Taxonomy labels"
STRONGREJECT_VLLM_URL=http://localhost:$PORT/v1 uv run python -m utils.heuristic.ta_taxonomy --run_dir "$RUN_DIR" \
    > "$LOG_DIR/${TS}_taxonomy.log" 2>&1
kill $SERVER; wait $SERVER 2>/dev/null || true; trap - EXIT

log "Importance"
unset HF_HUB_CACHE  # the sentence encoder lives in the persistent HF cache
CUDA_VISIBLE_DEVICES=$GPU uv run python -m utils.heuristic.ta_analyse --run_dir "$RUN_DIR" > "$LOG_DIR/${TS}_analyse.log" 2>&1
log "Done: $RUN_DIR/analysis"

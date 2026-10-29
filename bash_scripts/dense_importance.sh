#!/usr/bin/env bash
# After utils.heuristic.dense_resample: score the dense rollouts with the local gpt-oss judge,
# repair NaN scores, label (S_i, T_i) equivalence with the same judge, then compute
# Thought-Anchors importance with both "different" criteria (cosine, judge).
#
# Usage: bash bash_scripts/dense_importance.sh <gpu> <model_name> <prompts e.g. 806,963,1073> [repetitions=100]
set -euo pipefail

GPU=$1; MODEL=$2; PROMPTS=$3; N=${4:-100}
PORT=${VLLM_PORT:-8000}
export HF_HUB_CACHE="${FAST_HF_HUB_CACHE:-/mnt/weka/scratch/$USER/hf/hub}" HF_HUB_OFFLINE=1
LOG_DIR=logs/dense_importance; mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
log() { echo "[$(date '+%F %T')] $*"; }

CUDA_VISIBLE_DEVICES=$GPU uv run vllm serve openai/gpt-oss-20b --port "$PORT" --max-model-len 8192 \
    > "$LOG_DIR/${TS}_judge_server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
    kill -0 $SERVER 2>/dev/null || { log "Judge server died; see $LOG_DIR/${TS}_judge_server.log"; exit 1; }
    sleep 5
done
export STRONGREJECT_VLLM_URL=http://localhost:$PORT/v1
log "Judge ready"

log "Scoring rollouts"
uv run python -m utils.heuristic.compute_scores_rollouts --model_name "$MODEL" --quadrant --repetitions "$N" \
    --threads 128 > "$LOG_DIR/${TS}_score.log" 2>&1
NAN_FILES=$(uv run python -c "
import glob, pandas as pd
for f in sorted(glob.glob('results/$MODEL/dataset/quadrants/scored_full_resample_prompt*_rep_$N.csv')):
    if pd.read_csv(f, usecols=['strongreject_score']).strongreject_score.isna().any(): print(f)")
if [ -n "$NAN_FILES" ]; then
    log "Repairing NaNs"
    set -- $NAN_FILES
    uv run python -m utils.repair_scores --scored_csv "$1" --extra_csvs "${@:2}" --threads 128 \
        >> "$LOG_DIR/${TS}_score.log" 2>&1
fi

log "Equivalence labels"
uv run python -m utils.heuristic.sentence_equivalence --model_name "$MODEL" --prompts "$PROMPTS" \
    --repetitions "$N" --threads 128 > "$LOG_DIR/${TS}_equivalence.log" 2>&1
kill $SERVER; wait $SERVER 2>/dev/null || true; trap - EXIT

log "Importance"
unset HF_HUB_CACHE  # the sentence encoder lives in the persistent HF cache, not the fast weights copy
uv run python -m utils.heuristic.counterfactual_importance --model_name "$MODEL" --repetitions "$N" \
    --prompts "$PROMPTS" --equivalence "results/counterfactual_importance/$MODEL/equivalence_rep$N.csv" \
    > "$LOG_DIR/${TS}_importance.log" 2>&1
log "Done"

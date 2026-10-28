#!/usr/bin/env bash
# Score quadrant rollouts with the StrongREJECT rubric, judged by a local GPT-OSS-20B vLLM
# server (same judge and settings as run_pipeline.sh), then repair any NaN scores.
# Already-scored files are skipped, so re-running resumes.
#
# Usage: bash bash_scripts/score_quadrants.sh <gpus e.g. 0,1> <model_name> [model_name ...]
set -euo pipefail

GPUS=$1; shift
N_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
PORT=${VLLM_PORT:-8000}
export HF_HUB_CACHE="${FAST_HF_HUB_CACHE:-/scratch/hf/hub}"
export HF_HUB_OFFLINE=1

LOG_DIR="logs/score_quadrants"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
VLLM_LOG="$LOG_DIR/${TS}_judge_server.log"

CUDA_VISIBLE_DEVICES=$GPUS uv run vllm serve openai/gpt-oss-20b --port "$PORT" \
    --data-parallel-size "$N_GPUS" --max-model-len 8192 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "Waiting for judge server (log: $VLLM_LOG)..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null; do
    kill -0 $VLLM_PID 2>/dev/null || { echo "Judge server died; see $VLLM_LOG"; exit 1; }
    sleep 5
done
echo "Judge server ready."
export STRONGREJECT_VLLM_URL="http://localhost:${PORT}/v1"

for MODEL_NAME in "$@"; do
    echo "=== Scoring $MODEL_NAME ==="
    uv run python -m utils.heuristic.compute_scores_rollouts --model_name "$MODEL_NAME" \
        --repetitions 10 --quadrant
    echo "=== Repairing NaN scores for $MODEL_NAME ==="
    # Only repair untracked (newly scored) files, so committed paper data is left untouched
    NAN_FILES=$(uv run python -c "
import glob, subprocess, pandas as pd
tracked = set(subprocess.run(['git', 'ls-files', 'results/$MODEL_NAME'], capture_output=True, text=True).stdout.split())
for f in sorted(glob.glob('results/$MODEL_NAME/dataset/quadrants/scored_full_resample_prompt*_rep_10.csv')):
    if f not in tracked and pd.read_csv(f, usecols=['strongreject_score']).strongreject_score.isna().any(): print(f)")
    for f in $NAN_FILES; do
        uv run python -m utils.repair_scores --scored_csv "$f" | grep -E "Found|Repaired|Failed" || true
    done
done
echo "Done at $(date)"

#!/usr/bin/env bash
# Score quadrant rollouts with the StrongREJECT rubric, judged by a local GPT-OSS-20B vLLM
# server (same judge and settings as run_pipeline.sh), then repair any NaN scores.
# Already-scored files are skipped, so re-running resumes.
#
# Usage: bash bash_scripts/score_quadrants.sh <gpus e.g. 0,1> <model_name> [model_name ...]
set -euo pipefail

GPUS=$1; shift
IFS=, read -ra GPU_LIST <<< "$GPUS"
BASE_PORT=${VLLM_PORT:-8000}
export HF_HUB_CACHE="${FAST_HF_HUB_CACHE:-/scratch/hf/hub}"
export HF_HUB_OFFLINE=1

LOG_DIR="logs/score_quadrants"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# One independent judge server per GPU (vLLM 0.10.1 --data-parallel-size crashes on
# gpt-oss MoE layers); clients are spread round-robin over the servers.
SERVER_PIDS=()
PORTS=()
for i in "${!GPU_LIST[@]}"; do
    PORT=$((BASE_PORT + i)); PORTS+=($PORT)
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} uv run vllm serve openai/gpt-oss-20b --port "$PORT" \
        --max-model-len 8192 > "$LOG_DIR/${TS}_judge_server_gpu${GPU_LIST[$i]}.log" 2>&1 &
    SERVER_PIDS+=($!)
done
trap 'kill ${SERVER_PIDS[@]} 2>/dev/null || true' EXIT

for i in "${!PORTS[@]}"; do
    echo "Waiting for judge server on port ${PORTS[$i]}..."
    until curl -sf "http://localhost:${PORTS[$i]}/health" > /dev/null; do
        kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null || { echo "Judge server on port ${PORTS[$i]} died; see $LOG_DIR/${TS}_judge_server_gpu${GPU_LIST[$i]}.log"; exit 1; }
        sleep 5
    done
done
echo "Judge servers ready on ports ${PORTS[*]}."
export STRONGREJECT_VLLM_URL="http://localhost:${BASE_PORT}/v1"  # used by repair step

for MODEL_NAME in "$@"; do
    echo "=== Scoring $MODEL_NAME ==="
    # One threaded client per judge server, on disjoint file shards (evaluate_dataset's
    # per-file process pools are CPU-bound on this many small files)
    N_CLIENTS=${N_CLIENTS:-${#PORTS[@]}}
    PIDS=()
    for k in $(seq 0 $((N_CLIENTS - 1))); do
        STRONGREJECT_VLLM_URL="http://localhost:${PORTS[$((k % ${#PORTS[@]}))]}/v1" \
            uv run python -m utils.heuristic.compute_scores_rollouts --model_name "$MODEL_NAME" \
            --repetitions 10 --quadrant --shard "$k/$N_CLIENTS" --threads "${THREADS:-192}" \
            > "$LOG_DIR/${TS}_${MODEL_NAME//\//_}_client${k}.log" 2>&1 &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || { echo "Scoring client $pid failed"; exit 1; }; done
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

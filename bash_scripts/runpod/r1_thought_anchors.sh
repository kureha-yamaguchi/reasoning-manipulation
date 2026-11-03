#!/usr/bin/env bash
# Runs ON the RunPod pod (via pod_bootstrap.sh, PIPELINE=thought_anchors): generation only for the
# Thought Anchors analysis of DeepSeek-R1 (AWQ) — ta_resample on 15 held-out prompts x 2 CoTs, then
# the prefill intervention. Scoring, stance labels and analysis run back on ARC.
# Env: MODEL, TP (set by pod_bootstrap.sh)
set -euo pipefail
LOG_DIR=logs/thought_anchors; mkdir -p "$LOG_DIR"
POINTS=results/thought_anchors/$MODEL/quadrant_points.json
PROMPTS=80,431,501,531,789,854,912,963,988,1200,2003,2049,2142,2213,2299
R1_TOKENIZER=$HF_HOME/hub/models--deepseek-ai--DeepSeek-R1/snapshots/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad
ARGS=(--model_name "$MODEL" --prompts "$PROMPTS" --points_json "$POINTS" --tensor_parallel_size "$TP")
if [ "$MODEL" = QuixiAI/DeepSeek-R1-AWQ ]; then ARGS+=(--tokenizer_name "$R1_TOKENIZER" --disable_prefix_caching); fi
log() { echo "[$(date '+%F %T')] $*"; }

# Resume an interrupted resample run if one exists
RUN=$(readlink -e "results/thought_anchors/$MODEL/latest" 2> /dev/null || true)
log "Thought Anchors rollouts ${RUN:+(resuming $RUN)}"
uv run python -m utils.heuristic.ta_resample "${ARGS[@]}" ${RUN:+--run_dir "$RUN"} > "$LOG_DIR/r1_resample.log" 2>&1
log "Prefill intervention"
uv run python -m utils.heuristic.ta_prefill "${ARGS[@]}" > "$LOG_DIR/r1_prefill.log" 2>&1
log "Generation done"

#!/usr/bin/env bash
# Thought Anchors (stage 4) + prefill (stage 5) for a model too large to share a GPU with the judge
# (gpt-oss-120b on 80 GB H100s): generate with the model split across both GPUs (TP=2, judges off),
# then judge on both GPUs. Resumes an existing ta_resample run directory.
# Usage: GPUS=1,3 bash bash_scripts/ta_stage_tp.sh <model> <prompts> <ta_run_dir> <log_dir>
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1; PROMPTS=$2; RUN=$3; LOG=$4
GPUS=${GPUS:-0,1}; G0=${GPUS%%,*}; G1=${GPUS##*,}
unset VLLM_PORT
# Make CUDA ordinals match nvidia-smi / Slurm GRES indices (on htc-g059 CUDA's default FASTEST_FIRST order
# differs, so CUDA_VISIBLE_DEVICES=3 pointed at another job's GPU). Matters when launched outside a Slurm step.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "[$(date '+%F %T')] $*"; }
judges_up() {  # one after the other: simultaneous start-up can fail model inspection
    local port g
    for i in 0 1; do
        g=$([ $i = 0 ] && echo "$G0" || echo "$G1"); port=800$i
        CUDA_VISIBLE_DEVICES=$g uv run vllm serve openai/gpt-oss-20b --port $port --max-model-len 8192 --max-num-seqs 1024 \
            > "$LOG/judge${i}_$1.log" 2>&1 &
        local pid=$!
        until curl -sf localhost:$port/health > /dev/null; do
            kill -0 $pid 2> /dev/null || { log "Judge on GPU $g failed to start; see $LOG/judge${i}_$1.log"; exit 1; }
            sleep 5
        done
    done
}
judges_down() { pkill -f "vllm serve openai/gpt-oss-20b --port 800[01]" || true; sleep 15; }
trap judges_down EXIT

if [ -z "${SKIP_GEN:-}" ]; then  # SKIP_GEN=1 when generation already finished (reuses its log)
    log "Stage 4: Thought Anchors generation, TP=2 on GPUs $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS uv run python -m utils.heuristic.ta_resample --model_name "$MODEL" --prompts "$PROMPTS" \
        --run_dir "$RUN" --tensor_parallel_size 2 > "$LOG/4_ta_gen_tp2.log" 2>&1
fi
log "Thought Anchors generation done; judging on both GPUs"
judges_up ta
JUDGE_URLS=http://localhost:8000/v1,http://localhost:8001/v1 ENCODER_GPU=$G0 CUDA_VISIBLE_DEVICES=$GPUS \
    bash bash_scripts/ta_judge_loop.sh "$RUN:$LOG/4_ta_gen_tp2.log" > "$LOG/4_ta_judge_loop_tp2.log" 2>&1
log "Thought Anchors done: $RUN/analysis"
judges_down

log "Stage 5: prefill generation, TP=2"
CUDA_VISIBLE_DEVICES=$GPUS uv run python -m utils.heuristic.ta_prefill --model_name "$MODEL" --prompts "$PROMPTS" \
    --tensor_parallel_size 2 > "$LOG/5_prefill_gen_tp2.log" 2>&1
PRUN=$(readlink -f "results/thought_anchors_prefill/$MODEL/latest")
judges_up prefill
STRONGREJECT_VLLM_URLS=http://localhost:8000/v1,http://localhost:8001/v1 uv run python -m utils.heuristic.ta_score \
    --run_dir "$PRUN" --threads 512 > "$LOG/5_prefill_score.log" 2>&1
uv run python -m utils.heuristic.ta_prefill_analyse --run_dir "$PRUN" --resample_run "$RUN" > "$LOG/5_prefill_analyse.log" 2>&1
log "Prefill done: $PRUN/analysis"
log "All stages done"

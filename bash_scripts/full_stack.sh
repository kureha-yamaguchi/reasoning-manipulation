#!/usr/bin/env bash
# The paper's full experiment stack for one more model on this node's two GPUs, stage by stage.
# Every stage reuses the pipeline the other models went through, with the same settings:
#   1  Sec 3-4 / Fig 3: baseline on all 2,446 prompts (5 CoTs x 5 answers), StrongREJECT scores,
#      quadrant filter, 10-rollout prefix resampling of every eligible prompt, rollout scores
#      (bash_scripts/r1_quadrants.sbatch with FULL_POOL=1)
#   2  Sec 5: filter datasets + run_pipeline.sh phases 2-7 (refusal directions, ablated models, layer
#      selection, final generation, scores). Must run alone: its free_gpus kills every GPU process.
#   3  Sec 5 capability evals (AIME-2025, GPQA-Diamond, BBEH-mini), base and CoT-direction ablated model
#   4  Thought Anchors: 15 eligible prompts x 2 CoTs, 100 rollouts per sentence; generation on GPU 1
#      while GPU 0 judges; then stance labels, taxonomy, both importance analyses
#   5  Prefill intervention on the same prompts
# Stages skip finished work where the underlying scripts do; START_STAGE resumes later stages.
#
# Usage (repo root, in tmux): MODEL=nvidia/AceReason-Nemotron-7B bash bash_scripts/full_stack.sh
set -euo pipefail
MODEL=${MODEL:?set MODEL}
START=${START_STAGE:-1}
LAYERS=${LAYERS:-11,13,15,17,19,21,23}  # as for R1-Distill-Qwen-7B (same architecture)
LOG=logs/full_stack/$(basename "$MODEL")_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG"
unset VLLM_PORT
log() { echo "[$(date '+%F %T')] $*"; }
stage() { [ "$START" -le "$1" ]; }
log "Full stack for $MODEL from stage $START; logs in $LOG; commit $(git rev-parse --short HEAD)"

if stage 1; then
    log "Stage 1: baseline, quadrants, prefix resampling"
    CUDA_VISIBLE_DEVICES=0,1 MODEL=$MODEL TP=2 FULL_POOL=1 bash bash_scripts/r1_quadrants.sbatch > "$LOG/1_quadrants.log" 2>&1
fi

if stage 2; then
    log "Stage 2: refusal directions (Sec 5)"
    uv run -m utils.filter_all_datasets --model_name "$MODEL" --scored_csv scored_train_harmful_prompts_cot5_out5.csv \
        --cot_lower_threshold 0.2 --cot_upper_threshold 0.8 \
        --baseline_lower_threshold 0.2 --baseline_upper_threshold 0.8 > "$LOG/2_filter.log" 2>&1
    bash bash_scripts/run_pipeline.sh --model_name "$MODEL" --layers "$LAYERS" --start_phase 2 > "$LOG/2_directions.log" 2>&1
fi

if stage 3; then
    log "Stage 3: capability evals"
    printf '\n\n' | EVAL_MODELS=$MODEL TP_SIZE=2 bash bash_scripts/run_evals.sh > "$LOG/3_evals.log" 2>&1
fi

judge_up() {  # gpt-oss judge on GPU 0, port 8000
    CUDA_VISIBLE_DEVICES=0 uv run vllm serve openai/gpt-oss-20b --port 8000 --max-model-len 8192 --max-num-seqs 1024 \
        > "$LOG/judge.log" 2>&1 &
    JUDGE=$!
    until curl -sf localhost:8000/health > /dev/null; do kill -0 $JUDGE || { log "Judge died"; exit 1; }; sleep 5; done
}

if stage 4 || stage 5; then
    PROMPTS=$(uv run python -c "
import os, random
from utils.heuristic.resample_quadrants import find_quadrant, load_scored_csv
from utils.heuristic.ta_resample import BASELINES
pts = find_quadrant(load_scored_csv([os.path.join('results', '$MODEL', 'dataset', f) for f in BASELINES]))
print(','.join(map(str, sorted(random.Random(0).sample(sorted(p['prompt_idx'] for p in pts), 15)))))" 2> "$LOG/4_prompts.err" | tail -1)
    log "Thought Anchors prompts (seeded 15 of the eligible): $PROMPTS"
    judge_up
    trap 'kill $JUDGE 2> /dev/null || true' EXIT
fi

if stage 4; then
    log "Stage 4: Thought Anchors"
    CUDA_VISIBLE_DEVICES=1 uv run python -m utils.heuristic.ta_resample --model_name "$MODEL" --prompts "$PROMPTS" \
        > "$LOG/4_ta_gen.log" 2>&1 &
    GEN=$!
    until grep -q "CoTs to generate in" "$LOG/4_ta_gen.log"; do kill -0 $GEN || { log "TA generation died"; exit 1; }; sleep 10; done
    RUN=$(grep -oE "CoTs to generate in \S+" "$LOG/4_ta_gen.log" | awk '{print $NF}')
    JUDGE_URLS=http://localhost:8000/v1 ENCODER_GPU=1 bash bash_scripts/ta_judge_loop.sh "$RUN:$LOG/4_ta_gen.log" \
        > "$LOG/4_ta_judge_loop.log" 2>&1
    wait $GEN
    log "Thought Anchors done: $RUN/analysis"
fi

if stage 5; then
    log "Stage 5: prefill"
    CUDA_VISIBLE_DEVICES=1 uv run python -m utils.heuristic.ta_prefill --model_name "$MODEL" --prompts "$PROMPTS" \
        > "$LOG/5_prefill_gen.log" 2>&1
    PRUN=$(readlink -f "results/thought_anchors_prefill/$MODEL/latest")
    STRONGREJECT_VLLM_URLS=http://localhost:8000/v1 uv run python -m utils.heuristic.ta_score --run_dir "$PRUN" \
        --threads 512 > "$LOG/5_prefill_score.log" 2>&1
    uv run python -m utils.heuristic.ta_prefill_analyse --run_dir "$PRUN" \
        --resample_run "$(readlink -f "results/thought_anchors/$MODEL/latest")" > "$LOG/5_prefill_analyse.log" 2>&1
    log "Prefill done: $PRUN/analysis"
fi
log "All stages done"

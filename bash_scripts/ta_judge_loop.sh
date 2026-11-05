#!/usr/bin/env bash
# Judge-side companion to ta_resample running elsewhere: scores new rollout files as they appear,
# and once a run's generation has finished (its log says so), labels stance and taxonomy, then runs
# both importance analyses and figures. Needs judge server(s) up (JUDGE_URLS, comma-separated).
#
# JUDGE_URLS may list several servers; each step uses whichever are up when it starts.
# Usage: JUDGE_URLS=http://localhost:8000/v1 bash bash_scripts/ta_judge_loop.sh <run_dir>:<gen_log> [...]
set -uo pipefail
URLS=${JUDGE_URLS:?set JUDGE_URLS}
LOG_DIR=logs/thought_anchors
log() { echo "[$(date '+%F %T')] $*"; }
declare -A DONE
while true; do
    pending=0
    for pair in "$@"; do
        RUN=${pair%%:*}; GEN=${pair#*:}
        [ -n "${DONE[$RUN]:-}" ] && continue
        pending=1
        [ -d "$RUN" ] || continue
        finished=0; pgrep -f "[t]a_resample.*$(jq -r .args.model_name "$RUN/metadata.json")" > /dev/null || finished=1
        if [ $finished = 1 ] && ! grep -q "Saved" "$GEN" 2> /dev/null; then  # generator died before saving anything
            log "ERROR: generation for $RUN exited without output (see $GEN)"; exit 1
        fi
        grep -q "Saved" "$GEN" 2>/dev/null || continue
        log "Scoring $RUN"
        STRONGREJECT_VLLM_URLS=$URLS uv run python -m utils.heuristic.ta_score --run_dir "$RUN" --threads 512 \
            >> "$LOG_DIR/$(basename "$RUN")_score.log" 2>&1
        if [ $finished = 1 ]; then
            log "Generation finished for $RUN: stance, taxonomy, analysis"
            STANCE_JUDGE_URLS=$URLS uv run python -m utils.heuristic.ta_stance --run_dir "$RUN" --threads 512 \
                > "$LOG_DIR/$(basename "$RUN")_stance.log" 2>&1
            STRONGREJECT_VLLM_URL=${URLS%%,*} uv run python -m utils.heuristic.ta_taxonomy --run_dir "$RUN" \
                > "$LOG_DIR/$(basename "$RUN")_taxonomy.log" 2>&1
            for e in cosine stance; do
                CUDA_VISIBLE_DEVICES=${ENCODER_GPU:-0} uv run python -m utils.heuristic.ta_analyse --run_dir "$RUN" --equivalence $e \
                    > "$LOG_DIR/$(basename "$RUN")_analyse_$e.log" 2>&1
                uv run python -m utils.heuristic.ta_plot --run_dir "$RUN" --equivalence $e >> "$LOG_DIR/$(basename "$RUN")_analyse_$e.log" 2>&1
            done
            log "Done $RUN"
            DONE[$RUN]=1
        fi
    done
    [ $pending = 0 ] && { log "All runs done"; break; }
    sleep 120
done

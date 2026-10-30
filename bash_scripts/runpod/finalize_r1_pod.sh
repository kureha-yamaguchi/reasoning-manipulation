#!/usr/bin/env bash
# After run_r1_on_runpod.sh reports DONE (results pulled, pod stopped): verify the local copy
# against the pod's results tarball, and terminate the pod (deleting its volume) only if every
# check passes. Otherwise the stopped pod and its volume are left alone.
#
# Usage: bash bash_scripts/runpod/finalize_r1_pod.sh <controller log> <pod state json> <model>
set -euo pipefail
CTRL_LOG=$1; STATE=$2; MODEL=$3
log() { echo "[$(date '+%F %T')] $*"; }

until grep -qE "DONE:|FAILED|EXIT" "$CTRL_LOG"; do sleep 60; done
grep -q "DONE:" "$CTRL_LOG" || { log "Run did not finish cleanly; not terminating"; tail -3 "$CTRL_LOG"; exit 1; }
TARBALL=$(grep -oE "tarball in \S+" "$CTRL_LOG" | tail -1 | cut -d' ' -f3)/r1_results.tgz
log "Verifying $TARBALL against results/$MODEL"

gzip -t "$TARBALL"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/r1_verify.XXXX")
tar xzf "$TARBALL" -C "$TMP"
diff -rq "$TMP/results/$MODEL/dataset" "results/$MODEL/dataset"
log "Tarball contents identical to the local results/$MODEL/dataset"

Q=results/$MODEL/dataset/quadrants
N_SCORED=$(ls "$Q"/scored_full_resample_prompt*_rep_10.csv | wc -l)
N_RAW=$(ls "$Q"/full_resample_prompt*_rep_10.csv | wc -l)
N_BASE=$(ls results/"$MODEL"/dataset/scored_*_cot5_out5.csv | wc -l)
log "Local: $N_RAW rollout files, $N_SCORED scored, $N_BASE scored baselines"
[ "$N_SCORED" -eq "$N_RAW" ] && [ "$N_RAW" -gt 0 ] && [ "$N_BASE" -eq 3 ] || { log "Counts incomplete; not terminating"; exit 1; }
LOGDIR=$(dirname "$TARBALL")
[ -s "$LOGDIR/bootstrap.log" ] && ls "$LOGDIR"/r1_quadrants/*/4_rollouts.log > /dev/null || { log "Pod logs missing locally; not terminating"; exit 1; }
rm -rf "$TMP"

log "All checks passed; terminating pod"
uv run python -m utils.runpod_pod terminate --state "$STATE"
sleep 10
uv run python -m utils.runpod_pod status --state "$STATE" || true
log "Finalised"

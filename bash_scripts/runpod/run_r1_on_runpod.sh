#!/usr/bin/env bash
# Run the full-precision DeepSeek-R1 quadrant pipeline on a RunPod 8xH200/H200 NVL/B200/B300 pod, from here.
#   1. wait (up to MAX_WAIT_H) for 8x H200/H200 NVL/B200/B300 secure-cloud stock, any region, then deploy (utils.runpod_pod)
#   2. start bash_scripts/runpod/pod_bootstrap.sh on the pod at this repo's HEAD (must be pushed)
#   3. every 15 min, rsync results/logs back to results/deepseek-ai/DeepSeek-R1 and logs/runpod/
#   4. on DONE: final rsync + verify the tarball, then stop the pod (volume kept until terminated
#      by hand); on FAILED: pull logs, stop the pod
# Spend is capped at CAP_USD by a watchdog on the pod itself, so it holds even if this dies.
#
# Usage (repo root, in tmux): CAP_USD=450 bash bash_scripts/runpod/run_r1_on_runpod.sh
# 4-bit AWQ on 4xH200: MODEL=QuixiAI/DeepSeek-R1-AWQ TP=4 COUNT=4 GPU='NVIDIA H200' VOLUME_GB=800 CAP_USD=250 bash ...
set -euo pipefail
CAP_USD=${CAP_USD:-450}
MAX_WAIT_H=${MAX_WAIT_H:-24}
POLL_S=${POLL_S:-10}  # on-demand capacity is first come, first served; free 8-GPU machines go fast
# First in stock wins; on Blackwell (B200/B300) pod_bootstrap.sh swaps in torch cu128
GPU=${GPU:-NVIDIA H200,NVIDIA H200 NVL,NVIDIA B200,NVIDIA B300 SXM6 AC}
MODEL=${MODEL:-deepseek-ai/DeepSeek-R1}
TP=${TP:-8}
COUNT=${COUNT:-8}
VOLUME_GB=${VOLUME_GB:-1400}
TS=$(date +%Y%m%d_%H%M%S)
STATE=results/runpod/${TS}_r1_pod.json
LOGS=logs/runpod/$TS
mkdir -p results/runpod "$LOGS" "results/$MODEL"
log() { echo "[$(date '+%F %T')] $*"; }

SHA=${GIT_SHA:-$(git rev-parse HEAD)}  # commit the pod checks out (must be on origin)
git fetch -q origin || log "git fetch failed; checking the last-fetched remote state"
git branch -r --contains "$SHA" | grep -q . || { log "HEAD $SHA is not pushed"; exit 1; }

if [ -n "${POD_STATE:-}" ]; then  # adopt an existing pod (e.g. after fixing a failed run); cap counts from its creation
    cp "$POD_STATE" "$STATE"
else
    uv run python -m utils.runpod_pod launch --gpu "$GPU" --count "$COUNT" --volume_gb "$VOLUME_GB" --state "$STATE" --max_wait_h "$MAX_WAIT_H" --poll_s "$POLL_S"
fi
COST=$(python3 -c "import json; print(json.load(open('$STATE'))['cost_per_hr'])")
CAP_SECONDS=$(python3 -c "import json, time; s = json.load(open('$STATE')); print(int($CAP_USD / $COST * 3600 - (time.time() - s['created'])))")
[ "$CAP_SECONDS" -gt 600 ] || { log "Less than 10 min of budget left"; exit 1; }
log "Pod $(python3 -c "import json; print(json.load(open('$STATE'))['pod_id'])") at \$$COST/h; cap \$$CAP_USD = $((CAP_SECONDS / 3600)) h"

read -r IP PORT < <(uv run python -m utils.runpod_pod ssh --state "$STATE")
SSH_OPTS=(-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ConnectTimeout=20 -p "$PORT")
pod() { ssh "${SSH_OPTS[@]}" "root@$IP" "$@"; }
until pod true 2> /dev/null; do sleep 10; done
log "SSH up: root@$IP -p $PORT"

pod "cat > /workspace/.rp_key && chmod 600 /workspace/.rp_key" < ~/.config/runpod/api_key
scp -q -P "$PORT" -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new bash_scripts/runpod/pod_bootstrap.sh "root@$IP:/workspace/"
# A background job started over ssh can hold the session open, so bound the call and verify instead
timeout 30 ssh -n "${SSH_OPTS[@]}" "root@$IP" "cd /workspace && rm -f DONE FAILED PULLED && GIT_SHA=$SHA CAP_SECONDS=$CAP_SECONDS MODEL=$MODEL TP=$TP FULL_POOL=${FULL_POOL:-} N_PROMPTS=${N_PROMPTS:-1000} setsid nohup bash pod_bootstrap.sh > bootstrap.log 2>&1 < /dev/null &" || true
pod "pgrep -f 'bash pod_bootstrap.sh' > /dev/null" || { log "Bootstrap did not start"; exit 1; }
log "Bootstrap started (commit $SHA)"

pull() {
    rsync -az -e "ssh ${SSH_OPTS[*]}" "root@$IP:/workspace/repo/results/$MODEL/" "results/$MODEL/" 2> /dev/null || true
    rsync -az -e "ssh ${SSH_OPTS[*]}" "root@$IP:/workspace/repo/logs/" "root@$IP:/workspace/"{bootstrap,download,watchdog}.log "$LOGS/" 2> /dev/null || true
}
stop() { uv run python -m utils.runpod_pod stop --state "$STATE"; }

n=0
while true; do
    sleep 300; n=$((n + 1))
    if pod "test -f /workspace/DONE" 2> /dev/null; then
        pull
        rsync -az -e "ssh ${SSH_OPTS[*]}" "root@$IP:/workspace/r1_results.tgz" "$LOGS/"
        tar tzf "$LOGS/r1_results.tgz" > /dev/null && pod "touch /workspace/PULLED"
        log "DONE: results in results/$MODEL, tarball in $LOGS"; stop; exit 0
    fi
    if pod "test -f /workspace/FAILED" 2> /dev/null; then
        pull; log "FAILED: see $LOGS/bootstrap.log"; stop; exit 1
    fi
    if ! pod true 2> /dev/null; then
        status=$(uv run python -m utils.runpod_pod status --state "$STATE")
        log "SSH unreachable; pod: $status"
        echo "$status" | grep -qE '"desiredStatus": "(EXITED|TERMINATED)"' && { log "Pod stopped (cap?)"; exit 1; }
    fi
    [ $((n % 3)) -eq 0 ] && { pull; log "Synced; $(pod 'tail -1 /workspace/bootstrap.log' 2> /dev/null)"; }
done

#!/usr/bin/env bash
# Run the full-precision DeepSeek-R1 quadrant pipeline on a RunPod 8xH200 (or H200 NVL) pod, from here.
#   1. wait (up to MAX_WAIT_H) for 8xH200 / 8xH200 NVL secure-cloud stock, any region, then deploy (utils.runpod_pod)
#   2. start bash_scripts/runpod/pod_bootstrap.sh on the pod at this repo's HEAD (must be pushed)
#   3. every 15 min, rsync results/logs back to results/deepseek-ai/DeepSeek-R1 and logs/runpod/
#   4. on DONE: final rsync + verify the tarball, then stop the pod (volume kept until terminated
#      by hand); on FAILED: pull logs, stop the pod
# Spend is capped at CAP_USD by a watchdog on the pod itself, so it holds even if this dies.
#
# Usage (repo root, in tmux): CAP_USD=450 bash bash_scripts/runpod/run_r1_on_runpod.sh
set -euo pipefail
CAP_USD=${CAP_USD:-450}
MAX_WAIT_H=${MAX_WAIT_H:-24}
GPU=${GPU:-NVIDIA H200,NVIDIA H200 NVL}  # Hopper only: the pinned env (torch cu126) lacks Blackwell kernels
TS=$(date +%Y%m%d_%H%M%S)
STATE=results/runpod/${TS}_r1_pod.json
LOGS=logs/runpod/$TS
mkdir -p results/runpod "$LOGS" results/deepseek-ai/DeepSeek-R1
log() { echo "[$(date '+%F %T')] $*"; }

SHA=$(git rev-parse HEAD)
git fetch -q origin && git branch -r --contains "$SHA" | grep -q . || { log "HEAD $SHA is not pushed"; exit 1; }

uv run python -m utils.runpod_pod launch --gpu "$GPU" --count 8 --state "$STATE" --max_wait_h "$MAX_WAIT_H"
COST=$(python3 -c "import json; print(json.load(open('$STATE'))['cost_per_hr'])")
CAP_SECONDS=$(python3 -c "print(int($CAP_USD / $COST * 3600))")
log "Pod $(python3 -c "import json; print(json.load(open('$STATE'))['pod_id'])") at \$$COST/h; cap \$$CAP_USD = $((CAP_SECONDS / 3600)) h"

read -r IP PORT < <(uv run python -m utils.runpod_pod ssh --state "$STATE")
SSH_OPTS=(-i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ConnectTimeout=20 -p "$PORT")
pod() { ssh "${SSH_OPTS[@]}" "root@$IP" "$@"; }
until pod true 2> /dev/null; do sleep 10; done
log "SSH up: root@$IP -p $PORT"

pod "cat > /workspace/.rp_key && chmod 600 /workspace/.rp_key" < ~/.config/runpod/api_key
scp -q -P "$PORT" -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new bash_scripts/runpod/pod_bootstrap.sh "root@$IP:/workspace/"
pod "cd /workspace && GIT_SHA=$SHA CAP_SECONDS=$CAP_SECONDS setsid nohup bash pod_bootstrap.sh > bootstrap.log 2>&1 < /dev/null &"
log "Bootstrap started (commit $SHA)"

pull() {
    rsync -az -e "ssh ${SSH_OPTS[*]}" "root@$IP:/workspace/repo/results/deepseek-ai/DeepSeek-R1/" results/deepseek-ai/DeepSeek-R1/ 2> /dev/null || true
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
        log "DONE: results in results/deepseek-ai/DeepSeek-R1, tarball in $LOGS"; stop; exit 0
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

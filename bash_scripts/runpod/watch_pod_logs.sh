#!/usr/bin/env bash
# Show the newest quadrant-pipeline phase log on a RunPod pod, refreshed every 30 s.
# Usage: bash bash_scripts/runpod/watch_pod_logs.sh <ip> <port>
IP=$1; PORT=$2
while true; do
    out=$(ssh -i ~/.ssh/id_ed25519 -o ConnectTimeout=20 -p "$PORT" "root@$IP" '
        f=$(ls -t /workspace/repo/logs/r1_quadrants/*/*.log 2>/dev/null | head -1)
        echo "${f:-no phase log yet}"
        [ -n "$f" ] && tail -c 20000 "$f" | tr "\r" "\n" | grep -v "^$" | tail -n 20 | cut -c1-180' 2>&1)
    clear; date '+%F %T'; echo "$out"
    sleep 30
done

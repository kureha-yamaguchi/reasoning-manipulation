#!/usr/bin/env bash
# Email milestones of a RunPod controller log (run_r1_on_runpod.sh) to ALERT_EMAIL, one mail per event.
# Usage: ALERT_EMAIL=you@example.org bash bash_scripts/runpod/email_alerts.sh <controller log>
set -uo pipefail
LOG=$1; TO=${ALERT_EMAIL:?set ALERT_EMAIL}
send() { tail -15 "$LOG" | grep -vE "in stock|deploy failed" | mail -s "[R1 RunPod] $1" "$TO"; }
tail -n0 -F "$LOG" | while read -r line; do
    case "$line" in
        *"Pod "*"/h; cap"*) send "pod acquired: ${line#*] }" ;;
        *"Bootstrap started"*) send "bootstrap started on the pod" ;;
        *"Bootstrap did not start"*|*"is not pushed"*|*"Less than 10 min"*) send "controller stopped: ${line#*] }"; exit ;;
        *"DONE:"*) send "DONE, results pulled, pod stopped"; exit ;;
        *"FAILED"*) send "FAILED: ${line#*] }"; exit ;;
        *"Pod stopped"*) send "pod stopped: ${line#*] }"; exit ;;
    esac
done

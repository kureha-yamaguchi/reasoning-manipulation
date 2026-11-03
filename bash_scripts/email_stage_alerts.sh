#!/usr/bin/env bash
# Email each stage line of a campaign driver log (lines starting "[YYYY-MM-DD hh:mm:ss]") to ALERT_EMAIL,
# and a failure email if the tmux pane running it shows "FAILED rc=" (the campaign exits on any error).
# Usage: ALERT_EMAIL=you@x SUBJECT="AceReason stack" bash bash_scripts/email_stage_alerts.sh <driver log> <tmux target>
set -uo pipefail
LOG=$1; PANE=$2; TO=${ALERT_EMAIL:?}; SUBJ=${SUBJECT:-campaign}
tail -n0 -F "$LOG" | while read -r line; do
    case "$line" in \[20*) echo "$line" | mail -s "[$SUBJ] ${line#*] }" "$TO" ;; esac
    case "$line" in *"All stages done"*) exit ;; esac
done &
TAIL=$!
while kill -0 $TAIL 2> /dev/null; do
    if tmux capture-pane -p -t "$PANE" 2> /dev/null | grep -q "FAILED rc"; then
        L=$(ls -td logs/full_stack/*/ | head -1)
        { tail -5 "$LOG"; echo; echo "Newest stage log ($(ls -t "$L"*.log | head -1)):"; tail -30 "$(ls -t "$L"*.log | head -1)"; } \
            | mail -s "[$SUBJ] FAILED" "$TO"
        kill $TAIL; exit
    fi
    sleep 60
done

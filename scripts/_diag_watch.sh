#!/usr/bin/env bash
# Poll the v100 e2e log for the diagnostic lines (raw token ids) or a result.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'RAW OUT|per-id|DECODE B|E2E OK|Error|Traceback' /data00/home/chenkailun.c/models/e2e.log 2>/dev/null")
  [ -n "$out" ] && { echo "$out"; break; }
  sleep 15
done

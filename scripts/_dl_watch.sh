#!/usr/bin/env bash
# Poll the v100 bf16 download log until DONE or an error surfaces.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 -o BatchMode=yes v100 \
    "grep -Eo 'DONE|Traceback|Error|No space' /data00/home/chenkailun.c/models/dl-3.8.log 2>/dev/null | tail -1")
  [ -n "$out" ] && { echo "download-status: $out"; break; }
  sleep 60
done

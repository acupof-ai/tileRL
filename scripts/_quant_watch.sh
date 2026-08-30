#!/usr/bin/env bash
# Poll the v100 quantization log until DONE or an error surfaces.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -Eo 'DONE:.*|Traceback|Error|No space|Killed|allocate memory' /data00/home/chenkailun.c/models/quant-3.8.log 2>/dev/null | tail -1")
  [ -n "$out" ] && { echo "quant-status: $out"; break; }
  sleep 45
done

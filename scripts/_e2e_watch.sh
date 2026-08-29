#!/usr/bin/env bash
# Poll the v100 end-to-end log until it finishes, errors, or emits tok/s.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -Eo 'E2E OK|DECODE B=1.*|Traceback|Error|CUDA|out of memory|dtype mismatch|RuntimeError.*' /data00/home/chenkailun.c/models/e2e.log 2>/dev/null | tail -1")
  [ -n "$out" ] && { echo "e2e-status: $out"; break; }
  sleep 30
done
#!/usr/bin/env bash
# Poll the v100 27B 8-layer correctness run after the embedding fix.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'RAW OUT|decoded|CORRECT|Traceback|Error|mismatch' /data00/home/chenkailun.c/models/c8.log 2>/dev/null")
  [ -n "$out" ] && { echo "$out"; break; }
  sleep 20
done

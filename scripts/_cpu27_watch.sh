#!/usr/bin/env bash
# Poll the v100 CPU 27B 4-layer control run.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'RAW OUT|decoded|CORRECT|Traceback|Error|mismatch' /data00/home/chenkailun.c/models/cpu27.log 2>/dev/null")
  [ -n "$out" ] && { echo "$out"; break; }
  sleep 25
done

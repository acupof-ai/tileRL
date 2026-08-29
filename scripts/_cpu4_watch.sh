#!/usr/bin/env bash
# Poll the v100 CPU 4-layer control run.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'RAW OUT|decoded|CORRECT|Traceback|Error' /data00/home/chenkailun.c/models/cpu4.log 2>/dev/null")
  [ -n "$out" ] && { echo "$out"; break; }
  sleep 20
done

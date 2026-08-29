#!/usr/bin/env bash
# Poll the v100 sm70 8-layer no-fuse run.
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'RAW OUT|decoded|CORRECT|Traceback|Error|mismatch' /data00/home/chenkailun.c/models/nofuse.log 2>/dev/null")
  [ -n "$out" ] && { echo "$out"; break; }
  sleep 20
done

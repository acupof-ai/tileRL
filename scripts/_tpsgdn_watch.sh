#!/usr/bin/env bash
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'DECODE B=1|OUTPUT:|graph capture failed|E2E OK|Traceback|Error' /data00/home/chenkailun.c/models/tps_gdn.log 2>/dev/null")
  echo "$out" | grep -qE 'DECODE B=1|Traceback|Error' && { echo "$out"; break; }
  sleep 25
done

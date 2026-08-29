#!/usr/bin/env bash
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'DECODE B=1|E2E OK|OUTPUT:|Traceback|Error' /data00/home/chenkailun.c/models/tps.log 2>/dev/null")
  echo "$out" | grep -q "DECODE B=1" && { echo "$out"; break; }
  sleep 25
done

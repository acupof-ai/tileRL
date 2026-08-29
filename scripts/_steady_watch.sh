#!/usr/bin/env bash
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'steady median|STEADY OK|first 6 steps|Traceback|Error' /data00/home/chenkailun.c/models/steady.log 2>/dev/null")
  echo "$out" | grep -qE 'steady median|Traceback|Error' && { echo "$out"; break; }
  sleep 25
done

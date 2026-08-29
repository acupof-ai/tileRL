#!/usr/bin/env bash
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'DECODE B=1|OUTPUT:|Traceback|Error' /data00/home/chenkailun.c/models/tps_full.log 2>/dev/null")
  echo "$out" | grep -qE 'DECODE B=1|Traceback|Error' && {
    /usr/bin/ssh -o ConnectTimeout=15 v100 "grep -acE 'graph capture failed' /data00/home/chenkailun.c/models/tps_full.log | sed 's/^/capture-failed: /'; grep -aE 'OUTPUT:|DECODE B=1' /data00/home/chenkailun.c/models/tps_full.log"
    break
  }
  sleep 25
done

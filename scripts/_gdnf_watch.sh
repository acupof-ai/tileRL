#!/usr/bin/env bash
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'RAW OUT|logprob|CORRECT|graph capture failed|mismatch|Traceback|Error' /data00/home/chenkailun.c/models/../tilerl-v100/nohup.out 2>/dev/null; cat /private/tmp/gdnf.log 2>/dev/null")
  # actually read from the bg task output on the mac side is separate; poll v100 by re-run marker
  o2=$(/usr/bin/ssh -o ConnectTimeout=15 v100 "test -f /data00/home/chenkailun.c/models/gdnf.log && grep -aE 'RAW OUT|CORRECT' /data00/home/chenkailun.c/models/gdnf.log")
  [ -n "$o2" ] && { echo "$o2"; break; }
  sleep 25
done

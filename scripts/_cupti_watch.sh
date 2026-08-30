#!/usr/bin/env bash
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'CUPTI OK|Traceback|Error' /data00/home/chenkailun.c/models/cupti.log 2>/dev/null")
  echo "$out" | grep -qE 'CUPTI OK|Traceback|Error' && {
    /usr/bin/ssh -o ConnectTimeout=15 v100 "grep -aE 'total device|kernel |%$|nvjet|cutlass|gemv|fp4|gdn|attn|rmsnorm|elementwise|void ' /data00/home/chenkailun.c/models/cupti.log | head -30"
    break
  }
  sleep 25
done

#!/usr/bin/env bash
while :; do
  out=$(/usr/bin/ssh -o ConnectTimeout=15 v100 \
    "grep -aE 'PROFILE OK|Traceback|Error' /data00/home/chenkailun.c/models/prof.log 2>/dev/null")
  echo "$out" | grep -qE 'PROFILE OK|Traceback|Error' && {
    /usr/bin/ssh -o ConnectTimeout=15 v100 "grep -aE 'eager wall|paged_attention|linear_fp4|rmsnorm|gdn|linear_attn|^linear |rope|silu|embedding|op ' /data00/home/chenkailun.c/models/prof.log"
    break
  }
  sleep 25
done

#!/usr/bin/env bash
while :; do
  done=$(/usr/bin/ssh -o ConnectTimeout=15 v100 "grep -aE 'CORRECT|Traceback|Error' /data00/home/chenkailun.c/models/cap8.log 2>/dev/null")
  [ -n "$done" ] && {
    /usr/bin/ssh -o ConnectTimeout=15 v100 "grep -acE 'graph capture failed' /data00/home/chenkailun.c/models/cap8.log | sed 's/^/capture-failed-lines: /'; grep -aE 'RAW OUT|CORRECT' /data00/home/chenkailun.c/models/cap8.log"
    break
  }
  sleep 25
done

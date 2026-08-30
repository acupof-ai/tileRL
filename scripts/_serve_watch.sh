#!/usr/bin/env bash
# Wait for the v100 27B server to be ready, then verify landing + chat routes.
while :; do
  h=$(/usr/bin/ssh -o ConnectTimeout=15 v100 "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null")
  [ "$h" = "200" ] && break
  err=$(/usr/bin/ssh -o ConnectTimeout=15 v100 "grep -aE 'Traceback|Error' /data00/home/chenkailun.c/models/serve.log 2>/dev/null | tail -1")
  [ -n "$err" ] && { echo "server-error: $err"; exit 0; }
  sleep 20
done
land=$(/usr/bin/ssh -o ConnectTimeout=15 v100 "curl -s http://127.0.0.1:8000/ 2>/dev/null | grep -c 'Open the playground'")
chat=$(/usr/bin/ssh -o ConnectTimeout=15 v100 "curl -s http://127.0.0.1:8000/chat 2>/dev/null | grep -c 'setMode'")
echo "server ready: landing=$land chat=$chat"

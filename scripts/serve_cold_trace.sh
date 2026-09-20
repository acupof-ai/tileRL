#!/usr/bin/env bash
# Per-arm cold-tier sampler. Polls the serve /health and appends one line per
# poll to its trace file. Lifecycle-bound to the serve pid: it exits the instant
# that serve is gone (or a zombie), so it can never outlive one boot and keep
# writing the next arm's data into this arm's file. No iteration cap — the pid
# binding is the termination condition. A run of TRACE_MAX_FAILS consecutive
# unreadable /health answers (wrong port, malformed body) is fatal too: the pid
# can be alive while the server is wedged, and spinning out an empty trace is
# the silent failure this sampler exists to prevent.
#
#   serve_cold_trace.sh <out_file> <serve_pid> [poll_s]
# endpoint from LIVENESS_BASE (exported by serve_h20.sh); default :8000.
set -uo pipefail

OUT=${1:?"usage: serve_cold_trace.sh <out_file> <serve_pid> [poll_s]"}
SPID=${2:?"serve pid to track is required"}
EVERY=${3:-10}
BASE=${LIVENESS_BASE:-http://127.0.0.1:8000}
MAX_FAILS=${TRACE_MAX_FAILS:-30}

trap 'exit 0' TERM INT
# Write the requested file itself: a direct caller gets the trace without an
# outer redirect; the supervisor truncates the file before each boot.
exec >> "$OUT" 2>&1

# kill -0 reads a zombie alive; require a live, non-zombie pid.
serve_live() {
  local st
  st=$(ps -o stat= -p "$SPID" 2>/dev/null) || return 1
  [[ "$st" != Z* ]]
}

fails=0
while serve_live; do
  body=$(curl -s -m 3 "$BASE/health" 2>/dev/null)
  line=$(printf '%s' "$body" | python3 -c '
import json, sys, time
try:
    s = json.load(sys.stdin)["stats"]
except Exception:
    raise SystemExit
priv = s.get("kv_cold_bytes", 0) / 2**30
ssd = s.get("kv_cold_ssd_bytes", 0) / 2**30
sh = s.get("kv_cold_shared_bytes", 0) / 2**30
shs = s.get("kv_cold_shared_ssd_bytes", 0) / 2**30
print("%d priv=%.3f ssd=%.3f shared=%.3f shared_ssd=%.3f TOTAL=%.3f fin=%d decfwd=%d tok=%d" % (
    time.time(), priv, ssd, sh, shs, priv + ssd + sh + shs,
    s.get("finished", 0), s.get("decode_forwards", 0), s.get("tokens_generated", 0)))
' 2>/dev/null)
  if [ -n "$line" ]; then
    fails=0
    echo "$line"
  else
    fails=$((fails + 1))
    if [ "$fails" -ge "$MAX_FAILS" ]; then
      echo "$(date +%s) TRACE_FATAL: $fails consecutive unreadable health from $BASE; sampler exiting" >&2
      exit 3
    fi
  fi
  sleep "$EVERY"
done

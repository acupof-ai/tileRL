#!/usr/bin/env bash
# Per-arm cold-tier sampler. Polls /health and appends one line per poll to a
# per-arm trace. Lifecycle-bound to the serve pid: it exits the instant that
# serve is gone (or a zombie), so it can never outlive one boot and keep writing
# the next arm's data into the previous arm's file. No iteration cap — the pid
# binding is the termination condition, replacing a fixed loop that died silent
# mid-matrix (2026-09-20, 1200x10s exited between arms).
#
#   serve_cold_trace.sh <out_file> <serve_pid> [poll_s]
set -uo pipefail

OUT=${1:?"usage: serve_cold_trace.sh <out_file> <serve_pid> [poll_s]"}
SPID=${2:?"serve pid to track is required"}
EVERY=${3:-10}

trap 'exit 0' TERM INT

# kill -0 reads a zombie alive; require a live, non-zombie pid.
serve_live() {
  local st
  st=$(ps -o stat= -p "$SPID" 2>/dev/null) || return 1
  [[ "$st" != Z* ]]
}

while serve_live; do
  curl -s -m 3 localhost:8000/health 2>/dev/null | python3 -c '
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
' 2>/dev/null
  sleep "$EVERY"
done

#!/usr/bin/env bash
# Selftest for pod_sync.sh's tree-in-use check, on the SHIPPED text (POD_SYNC_EMIT_CHECK=1).
#
#   scripts/pod_sync_selftest.sh     # exits 0, prints PASS
#
# The check refuses to wipe a pod tree while a detached job's pid is alive. What it asserts:
#   * no marker                  -> proceeds
#   * live pid, matching start   -> REFUSES (exit 1, "refusing to wipe", marker kept)
#   * dead pid                   -> proceeds, marker removed
#   * live pid, stale start time -> treated as stale (pid reused), proceeds, marker removed
#   * ps unavailable -> REFUSES (a guard that cannot decide must not pass silently)
set -euo pipefail

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECK=$(POD_SYNC_EMIT_CHECK=1 bash "$ROOT/scripts/pod_sync.sh")
[ -n "$CHECK" ] || { echo "FAIL: empty check snippet" >&2; exit 1; }
cd "$TMP"

# 1. no marker -> proceeds
bash -c "$CHECK" || { echo "FAIL: no marker should pass" >&2; exit 1; }

# 2. live pid -> refuses, marker kept
sleep 30 & live=$!
echo "$live $(ps -o lstart= -p "$live" | tr -s ' ')" > .pod_running
if bash -c "$CHECK" 2>err; then
  echo "FAIL: a live pid should refuse the wipe" >&2; exit 1
fi
grep -q "refusing to wipe" err || { echo "FAIL: refusal message missing" >&2; exit 1; }
[ -f .pod_running ] || { echo "FAIL: refusal removed the marker" >&2; exit 1; }
disown "$live" 2>/dev/null || true; kill "$live" 2>/dev/null || true

# 3. dead pid -> proceeds, marker removed
sleep 0.1 & dead=$!; wait "$dead"
echo "$dead $(ps -o lstart= -p $$ | tr -s ' ')" > .pod_running
bash -c "$CHECK" || { echo "FAIL: a dead pid should pass as stale" >&2; exit 1; }
[ ! -f .pod_running ] || { echo "FAIL: stale marker not removed" >&2; exit 1; }

# 4. live pid with a stale start time (pid reused) -> stale, proceeds
sleep 30 & live=$!
echo "$live Wed Dec 31 23:59:59 1969" > .pod_running
bash -c "$CHECK" || { echo "FAIL: a reused pid with old start time should pass" >&2; exit 1; }
[ ! -f .pod_running ] || { echo "FAIL: marker not removed after stale line" >&2; exit 1; }
disown "$live" 2>/dev/null || true; kill "$live" 2>/dev/null || true

# 5. ps unavailable -> refuses loud (stat="" must not read as "stale")
mkdir -p "$TMP/nops"
printf '#!/bin/sh\necho ps: command not found >&2\nexit 127\n' > "$TMP/nops/ps"
chmod +x "$TMP/nops/ps"
sleep 30 & live=$!
echo "$live $(ps -o lstart= -p "$live" | tr -s ' ')" > .pod_running
if PATH="$TMP/nops:$PATH" bash -c "$CHECK" 2>err; then
  echo "FAIL: ps unavailable should refuse" >&2; exit 1
fi
grep -q "ps unavailable" err || { echo "FAIL: ps-unavailable message missing" >&2; exit 1; }
disown "$live" 2>/dev/null || true; kill "$live" 2>/dev/null || true

echo "PASS: pod_sync tree-in-use check"

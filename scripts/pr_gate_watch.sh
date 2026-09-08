#!/usr/bin/env bash
# Emit one line per change in a PR's gate state, then exit when it settles. For Monitor,
# which cannot carry a compound command.
#
#   scripts/pr_gate_watch.sh <pr> [poll-s] [rounds]
#
# Reads `--json name,state`, not the human table: the table's first column is
# `gate (macos-14)`, which `awk '{print $2}'` splits mid-name, so the STATUS column never
# reached the string being tested for "pending" and the first version reported a settle
# while both gates were IN_PROGRESS.
set -uo pipefail
PR=${1:?usage: pr_gate_watch.sh <pr> [poll-s] [rounds]}
POLL=${2:-45}
ROUNDS=${3:-80}
prev=""
for _ in $(seq 1 "$ROUNDS"); do
  # Only the `gate` checks decide a merge: GitGuardian passes in 1s, and reading its SUCCESS
  # as the gate's is a mistake made in this repo today.
  json=$(gh pr checks "$PR" --json name,state 2>/dev/null || true)
  now=$(printf '%s' "$json" | python3 -c '
import json,sys
try:
    rows = json.load(sys.stdin)
except Exception:
    print("unreadable"); raise SystemExit
g = [r for r in rows if r["name"].startswith("gate")]
print(" ".join(f"{r[chr(110)+chr(97)+chr(109)+chr(101)]}={r[chr(115)+chr(116)+chr(97)+chr(116)+chr(101)]}" for r in g) or "no gate checks")
' 2>/dev/null || echo unreadable)
  if [ "$now" != "$prev" ]; then
    echo "PR $PR: $now"
    prev=$now
  fi
  # Settled = at least one gate reported and none IN_PROGRESS/QUEUED/PENDING.
  case "$now" in
    unreadable|"no gate checks") ;;
    *IN_PROGRESS*|*QUEUED*|*PENDING*) ;;
    *) echo "PR $PR settled: $now"; exit 0 ;;
  esac
  sleep "$POLL"
done
echo "PR $PR: still unsettled after $ROUNDS rounds"

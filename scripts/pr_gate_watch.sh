#!/usr/bin/env bash
# Emit one line per change in a PR's gate state, then exit when it settles. For Monitor,
# which cannot carry a compound command.
#
#   scripts/pr_gate_watch.sh <pr> [poll-s] [rounds]
#
# Two instrument bugs already paid for here. (1) The first version parsed the human table with
# `awk '{print $2}'`, and the first column is `gate (macos-14)` -- awk split the name mid-field
# so the STATUS column never reached the string tested for "pending", and it reported a settle
# while both gates were IN_PROGRESS: a false green, in the direction of "go ahead".
# (2) The second used an inline `python3 -c` whose quoting broke inside this script and printed
# "unreadable" on a response it had read fine. `gh --jq` needs no second language.
set -uo pipefail
PR=${1:?usage: pr_gate_watch.sh <pr> [poll-s] [rounds]}
POLL=${2:-45}
ROUNDS=${3:-80}
prev=""
for _ in $(seq 1 "$ROUNDS"); do
  # Only the `gate` checks decide a merge: GitGuardian passes in seconds, and reading its
  # SUCCESS as the gate's is a mistake made in this repo today.
  now=$(gh pr checks "$PR" --json name,state \
        --jq '[.[] | select(.name | startswith("gate")) | "\(.name)=\(.state)"] | join(" ")' \
        2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ]; then
    now="unreadable"
  elif [ -z "$now" ]; then
    now="no gate checks yet"
  fi
  if [ "$now" != "$prev" ]; then
    echo "PR $PR: $now"
    prev=$now
  fi
  # Settled = at least one gate reported and none still running. "unreadable" and "none yet"
  # are NOT settled -- an absent reading is not a good reading.
  case "$now" in
    unreadable|"no gate checks yet") ;;
    *IN_PROGRESS*|*QUEUED*|*PENDING*) ;;
    *) echo "PR $PR settled: $now"; exit 0 ;;
  esac
  sleep "$POLL"
done
echo "PR $PR: still unsettled after $ROUNDS rounds"

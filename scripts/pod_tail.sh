#!/usr/bin/env bash
# Stream the interesting lines of a pod log as events. `tn exec` needs a compound command to
# reach inside the container, which a Monitor command cannot carry, so it lives in a file.
#
#   scripts/pod_tail.sh <log-basename> [grep-ere] [poll-s] [rounds]
#
# Prints only lines it has not printed before, so each new result is one event. Terminal
# states are in the default pattern: a filter matching only the happy path is silent through
# a crash, and silence reads exactly like still-running.
set -uo pipefail
LOG=${1:?usage: pod_tail.sh <log-basename> [ere] [poll-s] [rounds]}
ERE=${2:-'^seed |^pooled|^  mean|^  at cap|^reward |^  sd |Traceback|Error|error|FAILED|Killed|CUDA out of memory'}
POLL=${3:-30}
ROUNDS=${4:-120}
POD_NAME="${POD_NAME:-sglang-test}"
seen=0
for _ in $(seq 1 "$ROUNDS"); do
  out=$(tn exec "cid=\$(crictl ps -q --name $POD_NAME --state Running | head -1); \
        crictl exec \$cid bash -lc $(printf '%q' "grep -aE '$ERE' /work/$LOG 2>/dev/null | cat -n")" 2>/dev/null || true)
  n=$(printf '%s' "$out" | grep -c . || true)
  if [ "$n" -gt "$seen" ]; then
    printf '%s\n' "$out" | tail -n "$((n - seen))" | sed 's/^ *[0-9]*\t//'
    seen=$n
  fi
  sleep "$POLL"
done

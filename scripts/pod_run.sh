#!/usr/bin/env bash
# Run one job on one H20 card, with every pod gotcha we paid for on 2026-09-05
# already encoded. Nobody should hand-type this shape again.
#
#   scripts/pod_run.sh [--wait] <name> <card[,card...]> -- <command...>
#   scripts/pod_run.sh arms 6 -- python3 scripts/recapture_arms.py --steps 6
#   scripts/pod_run.sh tp2 0,6 -- torchrun --nproc_per_node=2 scripts/x.py
#
# It RETURNS AT LAUNCH, not at completion -- about two seconds in, after `started`. A caller
# looping over arms must poll /work/pod_run_<name>.out for POD_RUN_DONE_<name> or pass --wait,
# or the next arm lands on a card the previous one is still using: two 27B servers, one card,
# one port 8000 (2026-09-08). The waiting bash below is the POD-SIDE one, which is a different
# bash from the caller's and reaps the job; that is what the next line is about.
#
# What it encodes, each line a thing that actually went wrong:
#   * a bash parent that WAITS, so the job is reaped. `setsid nohup ... &` from a
#     shell that exits orphans the job to container PID 1, which here is
#     `sleep infinity` and never calls wait(): four jobs ended as permanent
#     zombies, one holding 28.2 GB long enough that another team nearly restarted
#     the container -- which would have killed a run on another card.
#   * logs under /work: it survives a container restart, and it is where
#     pod_sync and the runs already live. (Not a disk-space reason: /, /tmp and
#     /work are one filesystem -- same fsid, 263G free, 87% used.)
#   * the claim polls both shapes: $CMD may be the python on the card or a wrapper whose
#     python is a descendant, and each flag covers only one.
#   * an unclaimable job is KILLED, exit 4. An unclaimed card is what gets a container
#     restarted under someone else's run, so continuing past a refusal is the worse failure.
#   * the claim result is echoed by the CALLER, not only into /work/pod_run_<name>.out:
#     four launches in one night reported no claim line at all, and a silent success reads
#     exactly like a silent failure.
#   * a CUDA touch before claiming, so a short job has a device fd to be seen.
#   * release by name in a trap, so a crash does not leave the claim held.
#   * a zombie claim from a previous crash is released and re-acquired: kill -0
#     and /proc both call a zombie alive, and only `ps -o stat=` says Zs.
#   * refuse to start if a card already holds >64 MiB with no claim.
#   * on exit, print `ps -o stat=` for the job and nvidia-smi for the card, so
#     "it finished" is a reading rather than an assumption.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/pod_session.sh
. "$ROOT/scripts/pod_session.sh"
POD_NAME="${POD_NAME:-sglang-test}"
# One tree per session; overridable because pod_run_selftest.sh points it at a tempdir.
REMOTE_DIR="${REMOTE_DIR:-$(pod_session_tree "$ROOT")}"
AUPAI="${AUPAI:-/work/aupai}"
# Card ownership comes from aupai's card_assignment.json (read-only), not a
# local quota file: two files drift, and the drift is silent. pod_run claims
# any free card, so without this gate it takes another team's
# (2026-09-09: card 5 is aupai's, used for an hour before anyone noticed).
CARD_ASSIGNMENT_JSON="${CARD_ASSIGNMENT_JSON:-/work/aupai/runs/card_assignment.json}"
ORPHAN_MIB="${ORPHAN_MIB:-64}"
# seconds to poll for the job's device fd: a 27B load takes minutes to open the card
DEVICE_WAIT="${DEVICE_WAIT:-300}"

WAIT=0
[ "${1:-}" = --wait ] && { WAIT=1; shift; }
LEND_REF=""
[ "${1:-}" = --lend-ref ] && { LEND_REF="${2:-}"; shift 2; }
[ $# -ge 4 ] || { echo "usage: $0 [--wait] [--lend-ref <ref>] <name> <card[,card...]> -- <command...>" >&2
                  echo "  --wait, as the FIRST argument only. Without it this returns as soon as" >&2
                  echo "  the job is LAUNCHED, not when it finishes: poll" >&2
                  echo "  /work/pod_run_<name>.out for POD_RUN_DONE_<name>." >&2
                  echo "  --lend-ref <ref>: run a card not recorded as tileRL's in" >&2
                  echo "  card_assignment.json; the ref is a ledger record of a lend, echoed into the log." >&2
                  echo "  Exits: 3 orphan card, 4 unclaimable, 5 name already live, 6 --wait timed out," >&2
                  echo "         7 card outside grant with no --lend-ref." >&2
                  exit 2; }
NAME=$1 CARD=$2; shift 2
[ "$1" = "--" ] || { echo "$0: expected -- before the command" >&2; exit 2; }
shift
# Each argument printf %q'd separately, so a quoted multi-word argument survives. `CMD="$*"`
# flattened argv and the runner's unquoted `setsid $CMD` re-split it, so
# `-- bash -c '<script>'` arrived as `bash -c` with no operand: `bash: -c: option requires an
# argument`, exit 2, card claimed, nothing running, and the caller saw `started` and exit 0.
CMD=$(printf '%q ' "$@")

# The runner is built by sourcing this file with POD_RUN_EMIT_RUNNER=1, so a selftest gets
# the shipped text instead of a hand-kept copy: the previous selftest inlined its own
# runner and would have passed against any bug in this one.

pod_exec() {
  tn exec "cid=\$(crictl ps -q --name $POD_NAME --state Running | head -1); \
           [ -n \"\$cid\" ] || { echo 'pod: container not Running' >&2; exit 1; }; \
           crictl exec \$cid bash -lc $(printf '%q' "$1")"
}

# The runner, assembled here and executed inside the container. `wait` is the
# whole point: this bash stays alive as the job's parent and reaps it.
read -r -d '' RUNNER <<RUNNER_EOF || true
set -uo pipefail
cd $REMOTE_DIR
# Before anything that can fail: a number is attributed to a sha or visibly \`unknown\`.
echo "pod_run: tree $REMOTE_DIR sha \$(cat $REMOTE_DIR/.synced_commit 2>/dev/null || echo unknown)"
[ -d /work/tl013 ] && export PATH=/work/tl013/bin:\$PATH
export TILELANG_CACHE_DIR=/work/tilelang_cache
export PYTHONPATH=$REMOTE_DIR/src:$REMOTE_DIR/packages/tilerl-kernels/src
export TILERL_TARGET=\${TILERL_TARGET:-cuda} CUDA_VISIBLE_DEVICES=$CARD
export TILERL_QWEN38_SOURCE=\${TILERL_QWEN38_SOURCE:-/work/Qwen3.8-27B-NVFP4}
export REMOTE_DIR=$REMOTE_DIR

# Quota: refuse a card not recorded as tileRL's in aupai's card_assignment.json
# (read-only). pod_run claims any free card, so without this it silently takes
# another team's. --lend-ref is the escape hatch, echoed so the lend is auditable.
# Fail closed: a missing/unreadable file or classifier refuses every card.
for c in ${CARD//,/ }; do
  if [ -z "$LEND_REF" ]; then
    owner=\$(python3 $REMOTE_DIR/scripts/card_owner.py \$c "$CARD_ASSIGNMENT_JSON" 2>/dev/null || echo nofile)
    case "\$owner" in
      ours) ;;
      theirs) echo "pod_run: card \$c is granted to another team per card_assignment.json; a lend needs a ledger record (--lend-ref)" >&2; exit 7;;
      nofile) echo "pod_run: card_assignment.json not readable at $CARD_ASSIGNMENT_JSON" >&2
              echo "  tileRL's grant is 0,1,3,6 (ckl 2026-09-08). If that grant still holds, bypass with --lend-ref <ledger record>" >&2
              exit 7;;
      *) echo "pod_run: card \$c has no tileRL ownership in card_assignment.json (unclassified -> refuse); a lend needs a ledger record (--lend-ref)" >&2; exit 7;;
    esac
  else
    echo "pod_run: card \$c not recorded as tileRL's; lend ref: $LEND_REF"
  fi
done

# per card, because \`-i 0,1\` returns a line per card and the -gt test needs one integer
for c in ${CARD//,/ }; do
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$c)
  held=\$(python3 $AUPAI/scripts/card_claim.py status 2>/dev/null | grep -c " \$c " || true)
  if [ "\$used" -gt $ORPHAN_MIB ] && [ "\$held" -eq 0 ]; then
    echo "pod_run: card \$c holds \${used} MiB with no claim -- an orphan. Refusing." >&2
    exit 3
  fi
done

release() { python3 $AUPAI/scripts/card_claim.py release --name tilerl-$NAME >/dev/null 2>&1 || true; }
trap release EXIT INT TERM

# A claim names a pid because the card's memory is held by a pid, so it dies when that pid
# does: a wrapper running N arms has N-1 windows where the claim reads STALE and the card
# reads ORPHAN while a later arm runs. The rule is therefore that each arm re-claims its own
# python pid as it starts, never the wrapper's. \`pod_run_claim\` is that call, exported so a
# multi-arm wrapper can invoke it per arm -- the arms run inside \$CMD, out of reach of the
# block below.
pod_run_claim() {  # pod_run_claim <pid> -- claim CARD for it, or kill it and exit 4
  # polled, because the fd opens minutes into a 27B load
  local pid=\$1 out i rc
  echo "pod_run: claim pending for \$pid, polling up to ${DEVICE_WAIT}s for a device fd"
  for i in \$(seq 1 $DEVICE_WAIT); do
    kill -0 \$pid 2>/dev/null || break
    # a wrapper's python is a descendant; a direct python is the pid itself.
    # Success is the EXIT CODE, not the word "claimed": when pod_run's own block below already
    # resolved to this same descendant, acquire is a no-op that says "claim reused, not
    # re-taken" and returns 0. A substring match missed that and killed the job at DEVICE_WAIT
    # -- 6 minutes of card 0 on 2026-09-07, with the claim held and the server healthy.
    out=\$(python3 $AUPAI/scripts/card_claim.py acquire --name tilerl-$NAME --cards $CARD \\
            --pid \$pid --wait-for-device 1 2>&1) && rc=0 || rc=\$?
    [ \$rc -eq 0 ] && { echo "pod_run: \$out"; return 0; }
    case "\$out" in
      *ZOMBIE*)    python3 $AUPAI/scripts/card_claim.py release --name tilerl-$NAME >/dev/null 2>&1 || true;;
    esac
    out=\$(python3 $AUPAI/scripts/card_claim.py acquire --name tilerl-$NAME --cards $CARD \\
            --pid \$pid --require-device 2>&1) && rc=0 || rc=\$?
    [ \$rc -eq 0 ] && { echo "pod_run: \$out"; return 0; }
    case "\$out" in
      *ZOMBIE*)    python3 $AUPAI/scripts/card_claim.py release --name tilerl-$NAME >/dev/null 2>&1 || true;;
    esac
    sleep 1
  done
  if kill -0 \$pid 2>/dev/null; then
    echo "pod_run: card_claim FAILED, killing \$pid: \$out" >&2
    kill -TERM \$pid 2>/dev/null; wait \$pid 2>/dev/null; exit 4
  fi
  echo "pod_run: job exited before it could claim: \$out" >&2
}
export -f pod_run_claim 2>/dev/null || true

# The job, under THIS shell so it is reaped. setsid detaches it from the exec
# session's terminal; the & + wait keeps this bash as its parent.
# PYTHONUNBUFFERED rather than a -u spliced into argv: CMD is often
# bash -c '... python3 ...', so there is no argv position for the flag, and the env var
# reaches every python in the tree including ones a wrapper spawns. Measured 2026-09-08 on
# two evals of the same binary: buffered left the log at 0 bytes after 43 MINUTES of a
# healthy run, unbuffered had 2720 bytes in 30 s. A block-buffered log makes a long job
# indistinguishable from a hung one for its whole duration -- two sessions nearly declared
# that run dead. No backticks anywhere in this comment: the heredoc below is UNQUOTED, so a
# backtick here is command substitution on the CALLER -- writing the flag name in backticks
# made the assembly run it and print "command not found", and the selftest's arm 0 caught it.
setsid env PYTHONUNBUFFERED=1 $CMD > /work/$NAME.log 2>&1 < /dev/null &
JOB=\$!
echo "pod_run: job pid \$JOB, log /work/$NAME.log"

# an unclaimed card gets a container restarted under someone else's run
pod_run_claim \$JOB

wait \$JOB; rc=\$?
echo "pod_run: exit \$rc  stat=\$(ps -o stat= -p \$JOB 2>/dev/null || echo reaped)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $CARD
echo "POD_RUN_DONE_$NAME rc=\$rc"
exit \$rc
RUNNER_EOF

# Hand the assembled runner to a selftest and stop, so the text under test is this one.
[ "${POD_RUN_EMIT_RUNNER:-0}" = 1 ] && { printf '%s\n' "$RUNNER"; exit 0; }

B64=$(printf '%s' "$RUNNER" | base64 | tr -d '\n')
# The WRAPPER is still detached through an exiting shell, and so still ends as a
# zombie under PID 1. That is fine and deliberate: it is a bash, it holds no CUDA
# context, and it has already reaped the job that did. Only the job must not orphan.
# A relaunch of a LIVE name corrupts the running one: /work/pod_run_$NAME.sh is a fixed path, and
# bash reads a script by byte offset, so `base64 -d >` truncating it under a wrapper still parked in
# pod_run_claim's poll makes that wrapper resume at a stale offset and execute a fragment
# (`line 73: 0: command not found` from a 76-line file containing no bare 0, 2026-09-08). Both
# wrappers also share the .out, which is why the pid line came out truncated. Refuse instead:
# a name is one launch at a time.
#
# `pgrep -f` is NOT usable here: the pattern travels inside the checking command's own argv, so
# pgrep matches the very shell asking the question and the guard can never pass. Match on the
# process's own comm+args via ps and drop this pid, so the test is about other processes.
live=$(pod_exec "ps -eo pid=,args= | awk -v self=\$\$ '\$1 != self && /bash \/work\/pod_run_$NAME\.sh/ {print \$1}' | head -3" 2>/dev/null || true)
live=$(printf '%s' "$live" | tr -d '\r' | tr '\n' ' ' | sed 's/  */ /g; s/^ //; s/ $//')
[ -z "$live" ] || { echo "$0: a wrapper for '$NAME' is still running (pid $live)." >&2
                    echo "  Relaunching would rewrite its script under it. Wait, or use another name." >&2
                    exit 5; }

pod_exec "echo $B64 | base64 -d > /work/pod_run_$NAME.sh && setsid bash /work/pod_run_$NAME.sh > /work/pod_run_$NAME.out 2>&1 < /dev/null & sleep 2; echo started"
# The claim line used to reach only /work/pod_run_<name>.out on the pod, so four launches in
# one night printed no claim result at all to their caller and a silent success read exactly
# like a silent failure. Surface it here: the claim decides whether the card is yours.
claim=$(pod_exec "sed -n 's/^pod_run: \(.*claimed.*\|.*card_claim.*\)/\1/p' /work/pod_run_$NAME.out | head -2" 2>/dev/null || true)
echo "pod_run: claim: ${claim:-not reported yet -- check /work/pod_run_$NAME.out}"
# The default is a LAUNCH, not a run: this returns while the job is still going. Callers that
# loop over arms must poll POD_RUN_DONE_<name> themselves or a second arm lands on the same
# card -- two 27B servers on one card and one port, 2026-09-08. Kept as the default because
# callers rely on it; --wait is the opt-in.
if [ "$WAIT" = 1 ]; then
  # capped, so a hung job ends the poll rather than the poll outliving the pod. The cap needs its
  # own exit code: the loop used to fall through on exhaustion into the summary grep, which prints
  # nothing and exits 0, so `--wait` on a hung job burned the full 2 h and then reported success --
  # the launcher's rc decoupled from the job's, which is the defect this file was just fixed for.
  # `set -o pipefail` does not catch it either: the grep|tail runs inside `bash -lc`, which has no
  # pipefail, so tail's 0 wins. 5 is "refused, still live", 6 is "waited, never finished".
  done_seen=0
  for _ in $(seq 1 "${WAIT_TICKS:-480}"); do
    pod_exec "grep -q POD_RUN_DONE_$NAME /work/pod_run_$NAME.out 2>/dev/null" && { done_seen=1; break; }
    sleep 15
  done
  [ "$done_seen" = 1 ] || { echo "$0: --wait gave up after $(( ${WAIT_TICKS:-480} * 15 ))s; $NAME has not written POD_RUN_DONE_$NAME." >&2
                            exit 6; }
  pod_exec "grep -h 'POD_RUN_DONE_\|pod_run: exit' /work/pod_run_$NAME.out | tail -2"
fi
echo "pod_run: $NAME on card $CARD; tail /work/$NAME.log, wrapper /work/pod_run_$NAME.out"

#!/usr/bin/env bash
# Run one job on one H20 card, with every pod gotcha we paid for on 2026-09-05
# already encoded. Nobody should hand-type this shape again.
#
#   scripts/pod_run.sh <name> <card[,card...]> -- <command...>
#   scripts/pod_run.sh arms 6 -- python3 scripts/recapture_arms.py --steps 6
#   scripts/pod_run.sh tp2 0,1 -- torchrun --nproc_per_node=2 scripts/x.py
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
ORPHAN_MIB="${ORPHAN_MIB:-64}"
# seconds to poll for the job's device fd: a 27B load takes minutes to open the card
DEVICE_WAIT="${DEVICE_WAIT:-300}"

[ $# -ge 4 ] || { echo "usage: $0 <name> <card[,card...]> -- <command...>" >&2; exit 2; }
NAME=$1 CARD=$2; shift 2
[ "$1" = "--" ] || { echo "$0: expected -- before the command" >&2; exit 2; }
shift
CMD="$*"

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
  local pid=\$1 out i
  echo "pod_run: claim pending for \$pid, polling up to ${DEVICE_WAIT}s for a device fd"
  for i in \$(seq 1 $DEVICE_WAIT); do
    kill -0 \$pid 2>/dev/null || break
    # a wrapper's python is a descendant; a direct python is the pid itself
    out=\$(python3 $AUPAI/scripts/card_claim.py acquire --name tilerl-$NAME --cards $CARD \\
            --pid \$pid --wait-for-device 1 2>&1) || true
    case "\$out" in
      *"claimed"*) echo "pod_run: \$out"; return 0;;
      *ZOMBIE*)    python3 $AUPAI/scripts/card_claim.py release --name tilerl-$NAME >/dev/null 2>&1 || true;;
    esac
    out=\$(python3 $AUPAI/scripts/card_claim.py acquire --name tilerl-$NAME --cards $CARD \\
            --pid \$pid --require-device 2>&1) || true
    case "\$out" in
      *"claimed"*) echo "pod_run: \$out"; return 0;;
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
setsid $CMD > /work/$NAME.log 2>&1 < /dev/null &
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
pod_exec "echo $B64 | base64 -d > /work/pod_run_$NAME.sh && setsid bash /work/pod_run_$NAME.sh > /work/pod_run_$NAME.out 2>&1 < /dev/null & sleep 2; echo started"
# The claim line used to reach only /work/pod_run_<name>.out on the pod, so four launches in
# one night printed no claim result at all to their caller and a silent success read exactly
# like a silent failure. Surface it here: the claim decides whether the card is yours.
claim=$(pod_exec "sed -n 's/^pod_run: \(.*claimed.*\|.*card_claim.*\)/\1/p' /work/pod_run_$NAME.out | head -2" 2>/dev/null || true)
echo "pod_run: claim: ${claim:-not reported yet -- check /work/pod_run_$NAME.out}"
echo "pod_run: $NAME on card $CARD; tail /work/$NAME.log, wrapper /work/pod_run_$NAME.out"

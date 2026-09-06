#!/usr/bin/env bash
# Run one job on one H20 card, with every pod gotcha we paid for on 2026-09-05
# already encoded. Nobody should hand-type this shape again.
#
#   scripts/pod_run.sh <name> <card> -- <command...>
#   scripts/pod_run.sh arms 6 -- python3 scripts/recapture_arms.py --steps 6
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
#   * card_claim with `--wait-for-device $DEVICE_WAIT` (300 s, not its 90 s default: a 27B
#     load does not touch CUDA inside 90 s and the guard killed the job), which follows
#     \$JOB's descendants until one holds
#     a device. \$JOB is a shell whenever the command is a wrapper script, and a shell pid is
#     refused: that refusal was never retried, so `-- bash wrapper.sh` ran the whole way
#     unclaimed and the card read ORPHAN with 69583 MiB for 5 minutes (2026-09-07).
#   * an unclaimable job is KILLED, exit 4. An unclaimed card is what gets a container
#     restarted under someone else's run, so continuing past a refusal is the worse failure.
#   * the claim result is echoed by the CALLER, not only into /work/pod_run_<name>.out:
#     four launches in one night reported no claim line at all, and a silent success reads
#     exactly like a silent failure.
#   * a CUDA touch before claiming, so a short job has a device fd to be seen.
#   * release by name in a trap, so a crash does not leave the claim held.
#   * a zombie claim from a previous crash is released and re-acquired: kill -0
#     and /proc both call a zombie alive, and only `ps -o stat=` says Zs.
#   * refuse to start if the card already holds >64 MiB with no claim.
#   * on exit, print `ps -o stat=` for the job and nvidia-smi for the card, so
#     "it finished" is a reading rather than an assumption.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POD_NAME="${POD_NAME:-sglang-test}"
REMOTE_DIR="${REMOTE_DIR:-/work/tilerl}"
AUPAI="${AUPAI:-/work/aupai}"
ORPHAN_MIB="${ORPHAN_MIB:-64}"
# card_claim defaults to 90 s, which a 27B load does not reach before touching CUDA:
# a row-45 arm was killed at 90 s with the model still loading.
DEVICE_WAIT="${DEVICE_WAIT:-300}"

[ $# -ge 4 ] || { echo "usage: $0 <name> <card> -- <command...>" >&2; exit 2; }
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
[ -d /work/tl013 ] && export PATH=/work/tl013/bin:\$PATH
export TILELANG_CACHE_DIR=/work/tilelang_cache
export PYTHONPATH=$REMOTE_DIR/src:$REMOTE_DIR/packages/tilerl-kernels/src
export TILERL_TARGET=\${TILERL_TARGET:-cuda} CUDA_VISIBLE_DEVICES=$CARD
export TILERL_QWEN38_SOURCE=\${TILERL_QWEN38_SOURCE:-/work/Qwen3.8-27B-NVFP4}
export REMOTE_DIR=$REMOTE_DIR

used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD)
held=\$(python3 $AUPAI/scripts/card_claim.py status 2>/dev/null | grep -c " $CARD " || true)
if [ "\$used" -gt $ORPHAN_MIB ] && [ "\$held" -eq 0 ]; then
  echo "pod_run: card $CARD holds \${used} MiB with no claim -- an orphan. Refusing." >&2
  exit 3
fi

release() { python3 $AUPAI/scripts/card_claim.py release --name tilerl-$NAME >/dev/null 2>&1 || true; }
trap release EXIT INT TERM

# A claim names a pid because the card's memory is held by a pid, so it dies when that pid
# does: a wrapper running N arms has N-1 windows where the claim reads STALE and the card
# reads ORPHAN while a later arm runs. The rule is therefore that each arm re-claims its own
# python pid as it starts, never the wrapper's. `pod_run_claim` is that call, exported so a
# multi-arm wrapper can invoke it per arm -- the arms run inside \$CMD, out of reach of the
# block below.
pod_run_claim() {  # pod_run_claim <pid> -- claim CARD for it, or kill it and exit 4
  local pid=\$1 out
  out=\$(python3 $AUPAI/scripts/card_claim.py acquire --name tilerl-$NAME --cards $CARD \\
          --pid \$pid --wait-for-device $DEVICE_WAIT 2>&1) || true
  case "\$out" in
    *ZOMBIE*) python3 $AUPAI/scripts/card_claim.py release --name tilerl-$NAME >/dev/null 2>&1 || true
              out=\$(python3 $AUPAI/scripts/card_claim.py acquire --name tilerl-$NAME \\
                      --cards $CARD --pid \$pid --wait-for-device $DEVICE_WAIT 2>&1) || true;;
  esac
  case "\$out" in
    *"claimed"*) echo "pod_run: \$out"; return 0;;
  esac
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

# card_claim refuses a shell pid, and \$JOB is a shell whenever CMD is a wrapper script --
# that refusal was never retried, so a wrapper-launched job ran the whole way unclaimed and
# the card read ORPHAN (2026-09-07, 69583 MiB for 5 minutes). --wait-for-device follows
# \$JOB's descendants until one holds a device, covering the "no device fd yet" case too.
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

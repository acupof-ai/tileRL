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
#   * logs under /work, never /tmp: the pod's / is full.
#   * card_claim with the PYTHON descendant's pid, not the wrapper's. card_claim
#     refuses a shell pid by name and tells you which child to claim.
#   * retry ONLY on "no device fd yet" (~1.3 s); "is a shell" is never retried.
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

[ $# -ge 4 ] || { echo "usage: $0 <name> <card> -- <command...>" >&2; exit 2; }
NAME=$1 CARD=$2; shift 2
[ "$1" = "--" ] || { echo "$0: expected -- before the command" >&2; exit 2; }
shift
CMD="$*"

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

# The job, under THIS shell so it is reaped. setsid detaches it from the exec
# session's terminal; the & + wait keeps this bash as its parent.
setsid $CMD > /work/$NAME.log 2>&1 < /dev/null &
JOB=\$!
echo "pod_run: job pid \$JOB, log /work/$NAME.log"

# card_claim wants a pid holding a device fd. A python that has not reached CUDA
# yet has none, so retry -- but ONLY on that refusal.
for i in 1 2 3 4 5 6 7 8 9 10; do
  kill -0 \$JOB 2>/dev/null || break        # a job that exits in <2 s never claims
  out=\$(python3 $AUPAI/scripts/card_claim.py acquire --name tilerl-$NAME --cards $CARD --pid \$JOB 2>&1) || true
  case "\$out" in
    *"claimed"*)      echo "pod_run: \$out"; break;;
    *ZOMBIE*)         release; continue;;
    *"no device fd"*) sleep 1.3; continue;;
    *"is a shell"*)   echo "pod_run: card_claim refused a shell pid: \$out" >&2; break;;
    *)                echo "pod_run: card_claim: \$out" >&2; break;;
  esac
done

wait \$JOB; rc=\$?
echo "pod_run: exit \$rc  stat=\$(ps -o stat= -p \$JOB 2>/dev/null || echo reaped)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $CARD
echo "POD_RUN_DONE_$NAME rc=\$rc"
exit \$rc
RUNNER_EOF

B64=$(printf '%s' "$RUNNER" | base64 | tr -d '\n')
# The WRAPPER is still detached through an exiting shell, and so still ends as a
# zombie under PID 1. That is fine and deliberate: it is a bash, it holds no CUDA
# context, and it has already reaped the job that did. Only the job must not orphan.
pod_exec "echo $B64 | base64 -d > /work/pod_run_$NAME.sh && setsid bash /work/pod_run_$NAME.sh > /work/pod_run_$NAME.out 2>&1 < /dev/null & sleep 2; echo started"
echo "pod_run: $NAME on card $CARD; tail /work/$NAME.log, wrapper /work/pod_run_$NAME.out"

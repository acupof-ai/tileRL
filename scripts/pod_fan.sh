#!/usr/bin/env bash
# Run N commands on the pod, one per GPU, in parallel. Each argument is a
# shell command; argument i gets CUDA_VISIBLE_DEVICES=i. Syncs once, then
# launches all of them detached and polls until every log is done.
#
#   scripts/pod_fan.sh 'cmd for gpu0' 'cmd for gpu1' ...
#
# Serialising A/Bs on one card was the iteration bottleneck: 3-5 minutes per
# arm, and a sweep of four arms is four round trips.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/work/tilerl}"
POD_NAME="${POD_NAME:-sglang-test}"
NAME="${NAME:-fan}"

"$ROOT/scripts/pod_sync.sh" >/dev/null   # one sync for all arms

pod_exec() {
  tn exec "cid=\$(crictl ps -q --name $POD_NAME --state Running | head -1); crictl exec \$cid bash -lc $(printf '%q' "$1")"
}

script="set -x"$'\n'
i=0
for cmd in "$@"; do
  script+="export TILELANG_CACHE_DIR=/work/tilelang_cache PYTHONPATH=$REMOTE_DIR/src:$REMOTE_DIR/packages/tilerl-kernels/src TILERL_TARGET=cuda"$'\n'
  script+="cd $REMOTE_DIR && CUDA_VISIBLE_DEVICES=$i bash -c $(printf '%q' "$cmd") > /work/${NAME}_$i.log 2>&1 &"$'\n'
  i=$((i + 1))
done
script+="wait"$'\n'
script+="echo DONE_${NAME}"$'\n'

b64=$(printf '%s' "$script" | base64 | tr -d '\n')
pod_exec "echo $b64 | base64 -d > /work/$NAME.sh; rm -f /work/$NAME.log; setsid nohup bash /work/$NAME.sh > /work/$NAME.log 2>&1 < /dev/null & sleep 1"
echo "launched $i arms on GPUs 0..$((i - 1))"
while :; do
  out=$(pod_exec "cat /work/$NAME.log" 2>/dev/null) || true
  case "$out" in *"DONE_$NAME"*) break;; esac
  sleep 20
done
for j in $(seq 0 $((i - 1))); do
  echo "=== GPU $j"
  pod_exec "grep -vi tilelang /work/${NAME}_$j.log | tail -12" || true
done

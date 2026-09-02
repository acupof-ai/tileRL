#!/usr/bin/env bash
# Sync this checkout to the V100 (sm70) host and run a command there.
#   scripts/v100.sh 'python3 scripts/ab_gemv_xh.py'          # sync, run, wait
#   scripts/v100.sh run <name> 'command'                     # sync, detach, poll
#
# /usr/bin/ssh, not ssh: the local shell wraps ssh in a function that is not
# available to non-interactive invocations. /usr/bin/python3 (torch 2.5.1+cu121)
# matches the 535 driver; the repo .venv does not. nvcc must be 12.4 — /usr/bin's
# is 11.8 and rejects -std=c++20.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${V100_HOST:-v100}"
DIR="${V100_DIR:-\$HOME/tilerl-v100}"
SSH=/usr/bin/ssh
ENV="export PATH=/usr/local/cuda-12.4/bin:\$PATH PYTHONPATH=$DIR/src:$DIR/packages/tilerl-kernels/src TILERL_TARGET=cuda TILELANG_CACHE_DIR=\$HOME/.tilelang_cache"

tar czf - --exclude=.venv --exclude=__pycache__ --exclude=.git --exclude='*.pyc' \
    --exclude='*.egg-info' -C "$ROOT" . 2>/dev/null \
  | $SSH "$HOST" "mkdir -p $DIR && cd $DIR && tar xzf -"

if [ "${1:-}" = run ]; then
  name="$2"; cmd="$3"
  $SSH "$HOST" "cd $DIR && $ENV && rm -f /tmp/$name.log && setsid nohup bash -c $(printf '%q' "$cmd; echo DONE_$name") > /tmp/$name.log 2>&1 < /dev/null & sleep 1"
  echo "launched $name; polling /tmp/$name.log"
  while :; do
    out=$($SSH "$HOST" "cat /tmp/$name.log" 2>/dev/null) || true
    case "$out" in *"DONE_$name"*) echo "$out"; exit 0;; esac
    sleep 20
  done
fi

[ $# -gt 0 ] && $SSH "$HOST" "cd $DIR && $ENV && $1"

#!/bin/bash
# Serve the 27B NVFP4 dense on the V100 for four concurrent 8k-context sessions,
# and restart it if it dies. Run detached ON THE V100 HOST:
#
#   setsid nohup scripts/serve_v100_dense.sh >/dev/null 2>&1 &
#
# Dense 4x8192 fits HBM with 4.86 GiB free after build (2048 auto pages vs the
# required 2048); sparse is wrong at this shape (k=128 attends ~2.2k of 8k and
# decodes slower), spec loses on sm70, and dense graph capture is auto-off
# (a failed capture poisons torch's allocator on sm70). Stability measured
# 2026-09-13 (docs/serve-v100.md): 4x7.4k concurrent prefill peaked at
# 27,196 MiB / 0 OOM; a 4-worker soak ran 234 turns with 0 errors and no restart.
#
# Distinct from serve_v100.sh, the older single-session spec-on 32k launcher.
#
# ponytail: bash loop, not a systemd unit -- no linger on this host.
set -u

ROOT=/data00/home/chenkailun.c
REPO=$ROOT/tilerl-v100
LOG=$ROOT/serve70_dense.log
MAX_RESTARTS=10        # then stay dead: a crash loop is a finding, not a hiccup
LOG_CAP=$((32 * 1024 * 1024))
PORT=8000

command -v flock >/dev/null || { echo "flock(1) not found; refusing unlocked" >&2; exit 2; }
exec 9>"$ROOT/.serve70_dense.lock"
flock -n 9 || { echo "another supervisor is already running" >&2; exit 1; }
cd "$REPO" || exit 1
export PATH=/usr/local/cuda-12.4/bin:$PATH
export TILERL_TARGET=cuda
export TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp
export PYTHONPATH=$REPO/src:$REPO/packages/tilerl-kernels/src
CKPT=$ROOT/models/Qwen3.8-27B-NVFP4
export TILERL_QWEN38_SOURCE=$CKPT
export TILERL_MESSAGES_RECORD=$ROOT/messages_requests.jsonl

child=
stopping=
trap 'stopping=1; if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; wait "$child"; fi; exit 143' TERM INT
for ((n = 0; n <= MAX_RESTARTS; n++)); do
  # An unbounded log once filled a disk; truncate past the cap before each boot.
  if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$LOG_CAP" ]; then : > "$LOG"; fi
  echo "serve70: tree $REPO sha $(cut -c1-10 "$REPO/.synced_commit" 2>/dev/null || git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) boot $n at $(date -Is)" >> "$LOG"
  started=$SECONDS
  "$ROOT/venv70/bin/python" -u -m tilerl.cli serve --model qwen38-27b \
      --host 0.0.0.0 --port $PORT \
      --slots 4 --max-batch 4 --max-ctx 8192 >> "$LOG" 2>&1 &
  child=$!
  wait "$child"; rc=$?; child=
  echo "=== exit rc=$rc after $((SECONDS - started))s at $(date -Is) ===" >> "$LOG"
  [ -n "$stopping" ] && exit 0
  sleep 5
done
echo "=== gave up after $MAX_RESTARTS restarts at $(date -Is) ===" >> "$LOG"

#!/usr/bin/env bash
# Hybrid V100 supervisor: serve the sparse+d1+decode-graph 27B, warm both modes,
# restart on unhealthy liveness, and stop fusing after a crash burst.
#
#   setsid nohup scripts/serve_hybrid_v100.sh >/dev/null 2>&1 &
#
# Everything host-specific is overridable by env; defaults suit the V100 box
# with the tree at $HOME/tilerl-v100-sse:
#   SERVE_ROOT ($HOME) SERVE_REPO ($ROOT/tilerl-v100-sse) SERVE_PORT (8000)
#   SERVE_PYTHON ($ROOT/venv70/bin/python) SERVE_CKPT_DIR SERVE_DRAFT
#   SERVE_COLD_SSD SERVE_LOG SERVE_LOCK
#   MAX_RESTARTS (10) RESTART_FUSE_MAX (5) RESTART_FUSE_WINDOW_S (600)
# The fuse: the FUSE_MAX-th restart inside FUSE_WINDOW_S trips it and the
# supervisor exits 2 with a marker -- a crash burst that fast is a finding, and
# restarting through it hides it. Restarts spaced further apart than the window
# age out and never count together.
#
# ponytail: bash loop, not a systemd unit -- linger is off on this box, so a
# --user unit dies with the ssh session.
set -u

ROOT=${SERVE_ROOT:-$HOME}
REPO=${SERVE_REPO:-$ROOT/tilerl-v100-sse}
PORT=${SERVE_PORT:-8000}
PYTHON=${SERVE_PYTHON:-$ROOT/venv70/bin/python}
LOG=${SERVE_LOG:-$ROOT/servehybridsse.log}
LOCK=${SERVE_LOCK:-$ROOT/.servehybridsse.lock}
FUSE_STATE=${SERVE_FUSE_STATE:-$ROOT/.servehybridsse.fuse}
CKPT=${SERVE_CKPT_DIR:-$ROOT/models/Qwen3.8-27B-NVFP4}
DRAFT=${SERVE_DRAFT:-$ROOT/mmlu-assets/model_mtp.safetensors}
COLD_SSD=${SERVE_COLD_SSD:-$ROOT/sparse_cold_128k.bin}
MAX_RESTARTS=${MAX_RESTARTS:-10}
RESTART_FUSE_MAX=${RESTART_FUSE_MAX:-5}
RESTART_FUSE_WINDOW_S=${RESTART_FUSE_WINDOW_S:-600}
READY_TRIALS=${SERVE_READY_TRIALS:-600}
LOG_CAP=$((32 * 1024 * 1024))

command -v flock >/dev/null || { echo "flock(1) not found; refusing unlocked" >&2; exit 2; }
exec 9>"$LOCK"
flock -n 9 || { echo "another supervisor is already running" >&2; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$REPO" || exit 1
export PATH=/usr/local/cuda-12.4/bin:$PATH
export TILERL_TARGET=cuda
mkdir -p "$ROOT/tmp"
export TMPDIR=$ROOT/tmp TMP=$ROOT/tmp TEMP=$ROOT/tmp
export PYTHONPATH=$REPO/src:$REPO/packages/tilerl-kernels/src
export TILERL_QWEN38_SOURCE=$CKPT
export TILERL_MESSAGES_RECORD=$ROOT/messages_requests.jsonl
export LIVENESS_BASE="http://127.0.0.1:$PORT"

child=; guard=; stopping=
trap 'stopping=1; [ -n "$guard" ] && { pkill -TERM -P "$guard" 2>/dev/null; kill -TERM "$guard" 2>/dev/null; }; pkill -TERM -f "serve_liveness.py $LOG" 2>/dev/null; pkill -TERM -f "serve_warmup_hybrid.py" 2>/dev/null; if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; wait "$child"; fi; exit 143' TERM INT

# Rolling restart window. A timestamp is appended only on a real restart, so a
# line is one restart: prune lines older than the window, trip at FUSE_MAX.
fuse_tripped() {
  local now cutoff
  now=$(date +%s)
  cutoff=$((now - RESTART_FUSE_WINDOW_S))
  [ -f "$FUSE_STATE" ] || return 1
  awk -v c="$cutoff" '$1 >= c' "$FUSE_STATE" > "$FUSE_STATE.tmp" && mv "$FUSE_STATE.tmp" "$FUSE_STATE"
  [ "$(wc -l < "$FUSE_STATE")" -ge "$RESTART_FUSE_MAX" ]
}

for ((n = 0; n <= MAX_RESTARTS; n++)); do
  if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$LOG_CAP" ]; then : > "$LOG"; fi
  if fuse_tripped; then
    echo "=== FUSE: $RESTART_FUSE_MAX restarts within ${RESTART_FUSE_WINDOW_S}s, staying down at $(date -Is) ===" >> "$LOG"
    echo "=== remove $FUSE_STATE to arm again ===" >> "$LOG"
    exit 2
  fi
  sha=$(cat "$REPO/.synced_commit" 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown)
  echo "servehybrid: tree $REPO sha ${sha:0:10} boot $n at $(date -Is)" >> "$LOG"
  started=$SECONDS
  "$PYTHON" -u -m tilerl.cli serve --model qwen38-27b \
      --host 0.0.0.0 --port "$PORT" \
      --slots 4 --max-batch 4 --max-ctx 131072 \
      --sparse-k 128 --sparse-min-tokens 8192 \
      --cold-format f16 --kv-cold-bytes 8589934592 \
      --cold-ssd-path "$COLD_SSD" --cold-ssd-bytes 8589934592 \
      --draft "$DRAFT" --depth 1 --decode-graph >> "$LOG" 2>&1 &
  child=$!
  ( for ((i = 1; i <= READY_TRIALS; i++)); do kill -0 $child 2>/dev/null || exit 1
      curl -sf -m 3 -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2; done
    echo "servehybrid: warmup start at $(date -Is)" >> "$LOG"
    "$PYTHON" "$SCRIPT_DIR/serve_warmup_hybrid.py" >> "$LOG" 2>&1
    echo "servehybrid: warmup done at $(date -Is)" >> "$LOG"
    "$PYTHON" "$SCRIPT_DIR/serve_liveness.py" "$LOG" "$LIVENESS_BASE" >> "$LOG" 2>&1
    lrc=$?
    echo "servehybrid: liveness exit $lrc at $(date -Is), killing pid $child" >> "$LOG"
    kill -TERM "$child" 2>/dev/null
    gone=0
    for k in $(seq 1 30); do
      ps -o stat= -p "$child" 2>/dev/null || { gone=1; break; }; sleep 1
    done
    [ "$gone" != 1 ] && { kill -9 "$child" 2>/dev/null; sleep 3; }
    st=$(ps -o stat= -p "$child" 2>/dev/null) && echo "servehybrid: WARN pid $child still present: $st" >> "$LOG"
    # Record held MiB after the pid is gone; informational only -- NOT a gate on
    # the GPU reading 0 (another process or a slow release is for ops to see).
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d " ")
    echo "servehybrid: post-kill GPU ${gpu}MiB at $(date -Is)" >> "$LOG"
    exit 12 ) &
  guard=$!
  wait "$child"; rc=$?; child=
  # A crash after readiness leaves this boot's guard blocked in liveness's 60s
  # sleep; kill its python children then the subshell before the next boot, or
  # the old liveness loop survives and two guards poll one new child.
  if [ -n "$guard" ]; then
    pkill -TERM -P "$guard" 2>/dev/null
    kill -TERM "$guard" 2>/dev/null
    wait "$guard" 2>/dev/null
  fi
  guard=
  ran=$((SECONDS - started))
  echo "=== exit rc=$rc after ${ran}s at $(date -Is) ===" >> "$LOG"
  [ -n "$stopping" ] && exit 0
  [ "$rc" = 0 ] && exit 0
  date +%s >> "$FUSE_STATE"
  sleep 5
done
echo "=== gave up after $MAX_RESTARTS restarts at $(date -Is) ===" >> "$LOG"
exit 1

#!/usr/bin/env bash
# H20 (sm90) serve supervisor: the sparse k128 + d1 + decode-graph 27B, warm both
# modes, restart on unhealthy liveness, and stop fusing after a crash burst.
#
# RUN IT THROUGH pod_run.sh, never as a hand launcher -- pod_run owns the card
# claim, the reaping parent, /work/tl013 on PATH, PYTHONPATH and the pod-synced
# tree this executes in:
#
#   scripts/pod_run.sh [--wait] [--lend-ref "<ref>"] h20serve <card> -- \
#       bash scripts/serve_h20.sh
#
# It is the sm90 analogue of serve_hybrid_v100.sh, with the V100-only parts
# removed: no ~/venv70 (python3 is /work/tl013, torch 2.11.0+cu129 -- never
# `uv run`, whose cu130 torch the 12.9 driver cannot load), no hard-coded
# ~/models or the cuda-12.4 nvcc PATH (the driver is 12.9), and no flock --
# pod_run's one-live-name refusal is the mutual-exclusion guard.
#
# Everything host-specific is overridable by env:
#   SERVE_PORT (8000) SERVE_PYTHON (python3) SERVE_LOG (/work/serve_h20.log)
#   SERVE_CKPT_DIR ($TILERL_QWEN38_SOURCE or /work/tilerl-ckpt/Qwen3.8-27B-NVFP4)
#   SERVE_DRAFT ($CKPT/model_mtp.safetensors)
#   SERVE_COLD_SSD (/work/sparse_cold_h20.bin; set "" to run with no spill tier)
#   SERVE_COLD_FORMAT (f16) SERVE_KV_COLD_BYTES / SERVE_COLD_SSD_BYTES (8 GiB)
#   SERVE_SPARSE_K (128) SERVE_SPARSE_MIN (8192)
#   SERVE_SLOTS (8) SERVE_BATCH (8) SERVE_CTX (131072) SERVE_DEPTH (1)
#   SERVE_DECODE_GRAPH (1; set 0 for the graph-off measurement arm)
#   MAX_RESTARTS (10) RESTART_FUSE_MAX (5) RESTART_FUSE_WINDOW_S (600)
#
# `--dry-run` prints the resolved serve argv and exits 0 without touching a GPU:
# the hermetic gate. The fuse: the FUSE_MAX-th restart inside FUSE_WINDOW_S trips
# it and the supervisor exits 2 with a marker -- a crash burst that fast is a
# finding, and restarting through it hides it. Restarts spaced further apart than
# the window age out and never count together.
#
# ponytail: bash loop, not a systemd unit -- the pod has no init manager for a
# job that lives and dies with one pod_run claim.
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# /work/tl013 is the maintained cu129 interpreter; pod_run already puts it on
# PATH, prepend it again so a hand-run dry-run resolves the same python.
[ -d /work/tl013/bin ] && export PATH=/work/tl013/bin:$PATH

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=${SERVE_REPO:-${REMOTE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}}
PORT=${SERVE_PORT:-8000}
PYTHON=${SERVE_PYTHON:-python3}
LOG=${SERVE_LOG:-/work/serve_h20.log}
FUSE_STATE=${SERVE_FUSE_STATE:-/work/.serve_h20.fuse}
CKPT=${SERVE_CKPT_DIR:-${TILERL_QWEN38_SOURCE:-/work/tilerl-ckpt/Qwen3.8-27B-NVFP4}}
DRAFT=${SERVE_DRAFT:-$CKPT/model_mtp.safetensors}
COLD_SSD=${SERVE_COLD_SSD-/work/sparse_cold_h20.bin}
COLD_FORMAT=${SERVE_COLD_FORMAT:-f16}
KV_COLD_BYTES=${SERVE_KV_COLD_BYTES:-8589934592}
COLD_SSD_BYTES=${SERVE_COLD_SSD_BYTES:-8589934592}
SPARSE_K=${SERVE_SPARSE_K:-128}
SPARSE_MIN=${SERVE_SPARSE_MIN:-8192}
SLOTS=${SERVE_SLOTS:-8}
BATCH=${SERVE_BATCH:-8}
CTX=${SERVE_CTX:-131072}
DEPTH=${SERVE_DEPTH:-1}
# sm90 decode graph on by default. SERVE_DECODE_GRAPH=0 passes the explicit
# --no-decode-graph force-off: merely omitting --decode-graph leaves the CLI arg
# at its default None, which engine _graph_on resolves to AUTO = ON on sm90/cuda
# (cli.py --decode-graph is store_const default None; the only off switch is
# --no-decode-graph, const=False).
DECODE_GRAPH=${SERVE_DECODE_GRAPH:-1}
MAX_RESTARTS=${MAX_RESTARTS:-10}
RESTART_FUSE_MAX=${RESTART_FUSE_MAX:-5}
RESTART_FUSE_WINDOW_S=${RESTART_FUSE_WINDOW_S:-600}
READY_TRIALS=${SERVE_READY_TRIALS:-600}
LOG_CAP=$((32 * 1024 * 1024))

# The cold spill lives on the pod's writable /work by default (the same fs the
# coldsmall probes used). /mnt/data02 exists but is root-0755 at the mount point;
# point SERVE_COLD_SSD at a writable subdir there to use the NVMe tier. An empty
# value runs the sparse engine with a host-only cold budget and no SSD spill.
COLD_ARGS=()
if [ -n "$COLD_SSD" ]; then
  COLD_ARGS=(--cold-format "$COLD_FORMAT" --kv-cold-bytes "$KV_COLD_BYTES"
            --cold-ssd-path "$COLD_SSD" --cold-ssd-bytes "$COLD_SSD_BYTES")
fi

# SERVE_DECODE_GRAPH=0 must pass --no-decode-graph explicitly: on sm90 the CLI's
# default (None) AUTO-enables capture, so omitting the flag would stay graph-on.
GRAPH_ARGS=(--no-decode-graph)
[ "$DECODE_GRAPH" != 0 ] && GRAPH_ARGS=(--decode-graph)
# One arm descriptor, logged on every boot line and printed by --dry-run so the
# reader of /work/serve_h20.log can self-certify which arm served without relying
# on a relayed command line.
GRAPH_WORD=off; [ "$DECODE_GRAPH" != 0 ] && GRAPH_WORD=on
ARM_DESC="depth=$DEPTH sparse_k=$SPARSE_K decode_graph=$GRAPH_WORD ctx=$CTX slots=$SLOTS"

SERVE_ARGV=("$PYTHON" -u -m tilerl.cli serve --model qwen38-27b
  --host 0.0.0.0 --port "$PORT"
  --slots "$SLOTS" --max-batch "$BATCH" --max-ctx "$CTX"
  --sparse-k "$SPARSE_K" --sparse-min-tokens "$SPARSE_MIN"
  ${COLD_ARGS[@]+"${COLD_ARGS[@]}"}
  --draft "$DRAFT" --depth "$DEPTH"
  ${GRAPH_ARGS[@]+"${GRAPH_ARGS[@]}"})

if [ "$DRY_RUN" = 1 ]; then
  printf '%s\n' "${SERVE_ARGV[@]}"
  echo "repo=$REPO ckpt=$CKPT log=$LOG cold_ssd=${COLD_SSD:-<disabled>}"
  echo "arm: $ARM_DESC"
  command -v "$PYTHON" >/dev/null || { echo "python not on PATH: $PYTHON" >&2; exit 2; }
  exit 0
fi

cd "$REPO" || exit 1
export TILERL_TARGET=${TILERL_TARGET:-cuda}
TMP=${SERVE_TMP:-/work/tmp}
mkdir -p "$TMP" 2>/dev/null && export TMPDIR=$TMP TMP=$TMP TEMP=$TMP
export PYTHONPATH=$REPO/src:$REPO/packages/tilerl-kernels/src${PYTHONPATH:+:$PYTHONPATH}
export TILERL_QWEN38_SOURCE=$CKPT
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
  echo "serve_h20: tree $REPO sha ${sha:0:10} boot $n $ARM_DESC at $(date -Is)" >> "$LOG"
  started=$SECONDS
  "${SERVE_ARGV[@]}" >> "$LOG" 2>&1 &
  child=$!
  ( for ((i = 1; i <= READY_TRIALS; i++)); do kill -0 $child 2>/dev/null || exit 1
      curl -sf -m 3 -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2; done
    echo "serve_h20: warmup start at $(date -Is)" >> "$LOG"
    "$PYTHON" "$SCRIPT_DIR/serve_warmup_hybrid.py" >> "$LOG" 2>&1
    echo "serve_h20: warmup done at $(date -Is)" >> "$LOG"
    "$PYTHON" "$SCRIPT_DIR/serve_liveness.py" "$LOG" "$LIVENESS_BASE" >> "$LOG" 2>&1
    lrc=$?
    echo "serve_h20: liveness exit $lrc at $(date -Is), killing pid $child" >> "$LOG"
    kill -TERM "$child" 2>/dev/null
    gone=0
    for k in $(seq 1 30); do
      ps -o stat= -p "$child" 2>/dev/null || { gone=1; break; }; sleep 1
    done
    [ "$gone" != 1 ] && { kill -9 "$child" 2>/dev/null; sleep 3; }
    st=$(ps -o stat= -p "$child" 2>/dev/null) && echo "serve_h20: WARN pid $child still present: $st" >> "$LOG"
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d " ")
    echo "serve_h20: post-kill GPU ${gpu}MiB at $(date -Is)" >> "$LOG"
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

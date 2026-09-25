#!/bin/bash
# Serve the V100 production arm (pure-sparse, W1024/R32, narrow draft query) under
# the bounded restart supervisor, so a fatal device error restarts the process
# instead of leaving a half-dead server answering /health 200.
#
#   setsid nohup scripts/run_serve_v100_prod_supervised.sh >/dev/null 2>&1 &
#
# The loop, the readiness wait, the liveness guard and the crash-burst fuse all
# live in serve_hybrid_v100.sh; this file only supplies the arm. Keeping one copy
# of that machinery is the point -- a second copy per arm is how the arms drift.
#
# Why the supervisor is needed at all: an illegal memory access poisons the CUDA
# context, so after it every later CUDA call fails. The engine now exits on that
# (Backend.device_alive -> fatal_device_exit, exit 11), but exit alone leaves the
# service down until something restarts it. errors/2026-09-25-a-dead-cuda-context-is-not-an-exception-type.md
#
# The served arm is the measurement set, not a default: pure sparse
# (--sparse-min-tokens 0), a 1024-token window and a 32-tick refresh cadence.
set -u

ROOT=${SERVE_ROOT:-$HOME}
REPO=${SERVE_REPO:-$ROOT/tilerl-v100-prod-5a0c54cc}
PORT=${SERVE_PORT:-8000}

# The 27B's draft head wants a true-width query on this card; the env var is read
# at draft construction, so it must be in the environment before the serve starts.
# Exported here rather than in the supervisor: it is a property of this arm.
export TILERL_DRAFT_TRUE_Q_WIDTH=1

# Word-split on purpose: serve_hybrid_v100.sh treats SERVE_ARGS as an argv list.
SERVE_ARGS="--model qwen38-27b --slots 4 --max-batch 4 \
  --sparse-k 128 --sparse-min-tokens 0 --scorer bounds \
  --decode-graph --depth 1 \
  --draft $ROOT/mmlu-assets/model_mtp.safetensors \
  --draft-attn-window-tokens 2048 --sparse-window-tokens 1024 --sparse-refresh-ticks 32 \
  --kv-cold-bytes 1073741824 --cold-ssd-path $ROOT/sparse_cold_128k.bin \
  --cold-ssd-bytes 8589934592 --cold-format f16"
export SERVE_ARGS

export SERVE_REPO=$REPO SERVE_PORT=$PORT
export SERVE_PYTHON=${SERVE_PYTHON:-$ROOT/venv70/bin/python}
export SERVE_LOG=${SERVE_LOG:-$ROOT/serve_prod_supervised.log}
export SERVE_LOCK=${SERVE_LOCK:-$ROOT/.serve_prod_supervised.lock}
export SERVE_FUSE_STATE=${SERVE_FUSE_STATE:-$ROOT/.serve_prod.fuse}

exec bash "$(cd "$(dirname "$0")" && pwd)/serve_hybrid_v100.sh"

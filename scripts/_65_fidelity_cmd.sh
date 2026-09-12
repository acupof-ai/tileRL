#!/usr/bin/env bash
# V100 (sm70) overnight fidelity stage — one 32k held span. Runs in cc's host
# tree on the V100 (plain ssh, no container, no /work). Copy into the queue
# tree as scripts/_65_fidelity_cmd.sh.
#
# Arms, each number printed + flushed to JSON as it lands (read in this order):
#   fullk_bounds  continuity: k = pages-8, MUST be KL ~0 / top1 ~1 on the sm70
#                 paged_attention kernel — if not, the rest is invalid on sm70.
#   dense scrambled  identity-vs-scrambled physical layout (table-content test)
#   bounds1024, bounds128, random128, window_only, oracle1024
#   set dump bounds1024 vs oracle1024 at chunks 0/8/63 x all source groups
set -x

cd "$(dirname "$0")/.."

# python is whatever the host tree already uses (cc's env); fall back to PATH.
PY=${FIDELITY_PY:-python}

# The V100 host has TWO nvcc: /usr/bin/nvcc is CUDA 11.8 (rejects -std=c++20,
# TileLang emits it for the sm70 paged kernel) and /usr/local/cuda/bin/nvcc is
# 12.4 (the login-shell one every other stage used). A non-login chainer puts
# /usr/bin first, so force the 12.x toolkit ahead when present.
if [ -x /usr/local/cuda/bin/nvcc ] && ! nvcc --version 2>/dev/null | grep -q "release 1[2-9]"; then
  export PATH=/usr/local/cuda/bin:$PATH
fi

export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$HOME/.tilelang_cache}
export PYTHONPATH=$PWD/scripts:$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda
export TILERL_CARD_LEND="tileRL overnight queue (tilerl-65 fidelity checks), coordinator tilerl-a3"
# sm70 V100: f32 attention IO and f32 recurrent state. The harness builds its
# LinearStatePool via precision.dtype("recurrent_state") (f32 on cuda) and the
# KV pool from the backend io dtype (f32 on sm70) — nothing here forces bf16.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Inputs cc supplies via env (sensible defaults; override at queue time):
#   TILERL_QWEN38_SOURCE  27B NVFP4 checkpoint dir on the V100 host
#   FIDELITY_CORPUS       directory holding held_32768.jsonl
#   FIDELITY_OUT          JSON results path
#   FIDELITY_GPU          visible index (default 0), SPAN (default 0)
: "${TILERL_QWEN38_SOURCE:?set TILERL_QWEN38_SOURCE to the 27B NVFP4 checkpoint}"
CORPUS=${FIDELITY_CORPUS:?set FIDELITY_CORPUS to the dir with held_32768.jsonl}
OUT=${FIDELITY_OUT:-$PWD/fidelity-checks-v100.json}
GPU=${FIDELITY_GPU:-0}
SPAN=${FIDELITY_SPAN:-0}

echo "fidelity V100 start $(date -u +%FT%TZ) root=$PWD src=$TILERL_QWEN38_SOURCE gpu=$GPU out=$OUT"

# watchdog: 2x the ~60 min six-arm estimate = 7200s hard cap
CUDA_VISIBLE_DEVICES=$GPU timeout 7200 "$PY" scripts/fidelity_checks.py \
  "$CORPUS" --span "$SPAN" --ctx 32768 --out "$OUT"
rc=$?
echo "fidelity V100 EXIT_$rc end $(date -u +%FT%TZ) out=$OUT"
exit $rc

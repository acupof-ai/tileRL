#!/usr/bin/env bash
# V100 (sm70) fidelity stage — one 32k held span, PRODUCTION engine path.
# Runs in cc's host tree on the V100 (plain ssh, no container, no /work).
#
# scripts/fidelity_engine.py builds TWO real engines over the same 27B model:
#   dense  build_engine(sparse_k=0)
#   sparse build_engine(sparse_k=K, scorer="bounds")
# and compares prefill logits at 256 seeded positions (KL/top1) plus 64 tokens
# of native greedy decode. No SparseForward subclass, no hand-built packed table
# — the old replay harness diverged on the live engine (KL 1.02 at full k).
#
# k=128 is the only sparse arm that fits a 32 GB V100: the production hot pool
# is slots*(n_groups*k + window + chunk); 4 groups at k=1024 is ~8 GiB K+V over
# the ~4.9 GiB post-weights headroom. Full-k continuity is the CPU-tiny gate
# (--tiny: KL 0 / top1 1 / greedy equal at k=pages), not a V100 row.
set -x

cd "$(dirname "$0")/.."

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
export TILERL_CARD_LEND="tileRL overnight queue (tilerl-65 fidelity), coordinator tilerl-a3"
# sm70 V100: f32 attention IO and f32 recurrent state; build_engine derives both
# from the backend/precision, nothing here forces bf16.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

: "${TILERL_QWEN38_SOURCE:?set TILERL_QWEN38_SOURCE to the 27B NVFP4 checkpoint}"
CORPUS=${FIDELITY_CORPUS:?set FIDELITY_CORPUS to the dir with held_32768.jsonl}
OUT=${FIDELITY_OUT:-$PWD/fidelity-engine-v100.json}
GPU=${FIDELITY_GPU:-0}
SPAN=${FIDELITY_SPAN:-0}
KS=${FIDELITY_KS:-128}

echo "fidelity V100 start $(date -u +%FT%TZ) root=$PWD src=$TILERL_QWEN38_SOURCE gpu=$GPU ks=$KS out=$OUT"

# dense 32k ~10 min + one k=128 sparse arm ~8 min on V100; 5400s hard cap.
CUDA_VISIBLE_DEVICES=$GPU timeout 5400 "$PY" scripts/fidelity_engine.py \
  "$CORPUS" --span "$SPAN" --ctx 32768 --ks "$KS" --out "$OUT"
rc=$?
echo "fidelity V100 EXIT_$rc end $(date -u +%FT%TZ) out=$OUT"
exit $rc

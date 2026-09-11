#!/usr/bin/env bash
set -x
cd "$(dirname "$0")/.."
export PATH=/work/tl013/bin:$PATH TILELANG_CACHE_DIR=/work/tilelang_cache
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda
export TILERL_CARD_LEND="ckl order 2026-09-11 all 8 cards (session tilerl-a3)"
export TILERL_QWEN38_SOURCE=/work/tilerl-ckpt/Qwen3.8-27B-NVFP4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=7 timeout 12600 /work/tl013/bin/python scripts/output_fidelity.py \
  /work/indexer-corpus /work/fidelity-65.json --ctxs 32768:8 16384:4
echo EXIT_$?

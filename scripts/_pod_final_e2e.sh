#!/usr/bin/env bash
# Dev-only: end-to-end re-measurement after the fusion + GDN + scheduler
# merges. GPU 0, slice4. Stamps /work/final_e2e.done.
set -euo pipefail
cd /work/tilerl
export PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=0 TILELANG_CACHE_DIR=/work/tilelang_cache
SLICE=/host/tc27-nvfp4-slice4

python3 scripts/profile_slice.py "$SLICE" --layers 4 --decode-ticks 30 --decode-graph --fuse \
  > /work/final_e2e_profile.log 2>&1
python3 scripts/bench_batch_decode.py "$SLICE" --layers 4 --batches 1,2,4,8 --fuse \
  > /work/final_e2e_batch.log 2>&1
echo FINAL_E2E_DONE > /work/final_e2e.done

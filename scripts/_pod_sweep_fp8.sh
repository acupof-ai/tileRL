#!/usr/bin/env bash
# Dev-only: run the fp8 prefill GEMM sweep on GPU 6, detached under setsid.
# Stamps /work/sweep_fp8.done on completion. Usage (pod):
#   setsid bash /work/tilerl/scripts/_pod_sweep_fp8.sh [variants...] &
set -euo pipefail
cd /work/tilerl
export PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES="${SWEEP_GPU:-0}" TILELANG_CACHE_DIR=/work/tilelang_cache
echo "launched $(date)" > /work/sweep_fp8.launched
python3 scripts/_sweep_fp8_prefill.py "$@" > /work/sweep_fp8.log 2>&1
echo "SWEEP_DONE $(date)" > /work/sweep_fp8.done

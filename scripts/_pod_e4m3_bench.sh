#!/usr/bin/env bash
# Dev-only: e4m3-scale final bench on the H20 pod.
# Phase 1 (contention-independent): CUDA parity + per-shape roofline.
# Phase 2 (low-contention window): slice2/slice4 graph decode + prefill-512.
set -e
cd /work/tilerl

PHASE="${1:-all}"

if [ "$PHASE" = "parity" ] || [ "$PHASE" = "all" ]; then
  CUDA_VISIBLE_DEVICES=1 TILERL_TARGET=cuda PYTHONPATH=src \
    python3 -m pytest tests/test_ops_parity.py -q > /work/parity_e4m3.log 2>&1
  echo PARITY_DONE > /work/parity_e4m3.done
fi

if [ "$PHASE" = "roofline" ] || [ "$PHASE" = "all" ]; then
  CUDA_VISIBLE_DEVICES=2 TILERL_TARGET=cuda PYTHONPATH=src \
    python3 scripts/bench_fp4_gemv.py /host/tc27-nvfp4-slice2 --layers 2 --ticks 10 \
    > /work/gemv_roof_e4m3.log 2>&1
  echo GEMV_DONE > /work/gemv_roof_e4m3.done
fi

if [ "$PHASE" = "profile" ] || [ "$PHASE" = "all" ]; then
  GPU="${PROFILE_GPU:-1}"
  CUDA_VISIBLE_DEVICES=$GPU TILERL_TARGET=cuda PYTHONPATH=src \
    python3 scripts/profile_slice.py /host/tc27-nvfp4-slice2 --layers 2 \
    --decode-graph --decode-ticks 30 > /work/final_e4m3_slice2.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU TILERL_TARGET=cuda PYTHONPATH=src \
    python3 scripts/profile_slice.py /host/tc27-nvfp4-slice4 --layers 4 \
    --decode-graph --decode-ticks 30 > /work/final_e4m3_slice4.log 2>&1
  echo PROFILE_DONE > /work/final_e4m3_profile.done
fi

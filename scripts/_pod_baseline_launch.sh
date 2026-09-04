#!/usr/bin/env bash
# Dev-only: Qwen3.8-27B baseline bench, detached (the silent 10+ min load outlasts tn exec's
# 5-min no-output timeout). Logs to /work/tilerl_baseline.log; stamps /work/tilerl_baseline.done.
set -e
cd /work/tilerl_baseline
setsid bash -c 'PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=7 TILELANG_CACHE_DIR=/work/tilelang_cache python3 scripts/bench_qwen38_baseline.py /data00/Qwen3.8-27B-NVFP4 > /work/tilerl_baseline.log 2>&1; echo DONE > /work/tilerl_baseline.done' </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pid $!"

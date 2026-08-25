#!/usr/bin/env bash
# Dev-only: re-profile + GEMV gap + batch decode measurement chain on the pod.
# GPU 0, slice4 (3 GDN + 1 FA = the 27B's 48:16 mix). Logs to /work/*.log.
set -euo pipefail
cd /work/tilerl
export PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=0 TILELANG_CACHE_DIR=/work/tilelang_cache
SLICE=/host/tc27-nvfp4-slice4

python3 scripts/profile_slice.py "$SLICE" --layers 4 --decode-ticks 30 > /work/reprofile_eager.log 2>&1
python3 scripts/profile_slice.py "$SLICE" --layers 4 --decode-ticks 30 --decode-graph > /work/reprofile_graph.log 2>&1
python3 scripts/bench_gemv_gap.py "$SLICE" --layers 4 > /work/gemv_gap.log 2>&1
python3 scripts/bench_batch_decode.py "$SLICE" --layers 4 > /work/batch_decode.log 2>&1
echo ALL_DONE > /work/measure.done

#!/usr/bin/env bash
# Dev-only: B>1 decode graph bench — eager vs graph arms, B=1/2/4/8, slice4.
# Run on the pod in an idle GPU window. Stamps /work/bgraph.done.
set -euo pipefail
cd /work/tilerl
export PYTHONPATH=src TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tilelang_cache
SLICE=/host/tc27-nvfp4-slice4
G="${1:?usage: _pod_bgraph_bench.sh <gpu>}"
CUDA_VISIBLE_DEVICES=$G python3 scripts/bench_batch_decode.py "$SLICE" \
  --layers 4 --batches 1,2,4,8 --fuse > /work/bgraph_eager.log 2>&1
CUDA_VISIBLE_DEVICES=$G python3 scripts/bench_batch_decode.py "$SLICE" \
  --layers 4 --batches 1,2,4,8 --fuse --decode-graph > /work/bgraph_graph.log 2>&1
echo BGRAPH_DONE > /work/bgraph.done

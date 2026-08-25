#!/usr/bin/env bash
# Dev-only: wait for GPU0 idle, then run the fusion A/B (decode graph, slice4).
# Launched under setsid; stamps /work/fuse_ab.done on completion.
set -euo pipefail
cd /work/tilerl
export PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=0 TILELANG_CACHE_DIR=/work/tilelang_cache
SLICE=/host/tc27-nvfp4-slice4
echo "launched $(date)" > /work/fuse_ab.launched

idle=0
for _ in $(seq 1 180); do  # 1h max: 20s between checks, confirm twice
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "$(date +%H:%M:%S) util=$u" >> /work/fuse_ab_wait.log
  if [ -n "$u" ] && [ "$u" -lt 50 ]; then
    sleep 3
    u2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if [ "$u2" -lt 50 ]; then idle=1; break; fi
  fi
  sleep 20
done
if [ "$idle" = 0 ]; then echo WAIT_TIMEOUT > /work/fuse_ab.done; exit 1; fi
echo "idle at $(date)" >> /work/fuse_ab_wait.log

python3 scripts/profile_slice.py "$SLICE" --layers 4 --decode-ticks 30 --decode-graph \
  > /work/fuse_base.log 2>&1
python3 scripts/profile_slice.py "$SLICE" --layers 4 --decode-ticks 30 --decode-graph --fuse \
  > /work/fuse_on.log 2>&1
echo FUSE_AB_DONE > /work/fuse_ab.done

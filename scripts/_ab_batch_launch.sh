#!/usr/bin/env bash
# Dev-only: launch the batched-decode A/B detached on the H20 pod (GPU 6).
# The 3-arm A/B has silent JIT phases (two new kernels) longer than tn
# exec's 5-min no-output timeout, so run detached and poll the done-stamp.
# Quiet-gated on GPU 6 (10-min fail-closed). Logs to /work/ab_batch.log.
set -e
cd /work/tilerl_p2_batch

setsid bash -c '
for i in $(seq 1 40); do
  out=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader -i 6 2>/dev/null) || { sleep 15; continue; }
  busy=$(echo "$out" | awk -F", " "NF>=2 && \$2+0 >= 10" | wc -l | tr -d " ")
  if [ "$busy" -eq 0 ]; then echo "quiet: $out" >&2; break; fi
  if [ "$i" -eq 40 ]; then echo "GPU 6 busy for 10min — aborting" >&2; exit 1; fi
  sleep 15
done
PYTHONPATH=src TILERL_TARGET=cuda CUDA_VISIBLE_DEVICES=6 TILELANG_CACHE_DIR=/work/tilelang_cache \
  BENCH_COMMIT=0518d76 python3 scripts/ab_batch_decode.py /host/tc27-nvfp4-slice4 --layers 4 \
  --arms shipped,ks1
echo $? > /work/ab_batch.done
' > /work/ab_batch.log 2>&1 </dev/null &
echo "LAUNCHED pid $!"

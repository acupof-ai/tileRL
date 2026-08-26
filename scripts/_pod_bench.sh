#!/usr/bin/env bash
# Sync + run a bench command on the pod on GPUs 6,7, quiet-gated.
# Usage: scripts/_pod_bench.sh 'PYTHONPATH=src python3 scripts/bench_smoke.py'
set -euo pipefail
BENCH_COMMIT="$(git rev-parse --short HEAD)"
# Quiet gate: GPUs 6,7 both <10% util (3 checks, 15s apart, 10min max).
gate='for i in $(seq 1 40); do
  out=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader -i 6,7 2>/dev/null) || { sleep 15; continue; }
  busy=$(echo "$out" | awk -F", " "NF>=2 && \$2+0 >= 10" | wc -l | tr -d " ")
  if [ "$busy" -eq 0 ]; then echo "quiet: $out" >&2; break; fi
  if [ "$i" -eq 40 ]; then echo "GPU 6,7 busy for 10min — running anyway" >&2; break; fi
  sleep 15
done'
scripts/pod_sync.sh "$gate && CUDA_VISIBLE_DEVICES=6,7 BENCH_COMMIT=$BENCH_COMMIT $1"

#!/usr/bin/env bash
# Dev-only: poll the pod for an idle GPU (<50% util, confirmed twice), print
# its index, and exit. Used to wait out other agents' sweeps before benching.
set -euo pipefail
for _ in $(seq 1 240); do  # 2h max: 30s between checks
  out=$(~/bin/pod 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader' 2>/dev/null || true)
  idle=$(echo "$out" | awk -F', ' '$2 < 50 {print $1}')
  if [ -n "$idle" ]; then
    sleep 3
    out2=$(~/bin/pod 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader' 2>/dev/null || true)
    idle2=$(echo "$out2" | awk -F', ' '$2 < 50 {print $1}')
    # pick a GPU idle in both checks
    for g in $idle2; do
      if echo "$idle" | grep -q "^$g$"; then echo "IDLE $g"; exit 0; fi
    done
  fi
  sleep 30
done
echo "TIMEOUT"
exit 1

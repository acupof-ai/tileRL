#!/usr/bin/env bash
# Run decode bench with SSD tier enabled (measures the GIL yield cost when SSD is on).
set -euo pipefail
rm -rf /tmp/ssd-bench
python3 scripts/bench_batch_decode.py /host/tc27-nvfp4-slice4 \
  --layers 4 --batches 1,8 --fuse --decode-graph \
  --card "${CARD:-3}" --device-name "NVIDIA H20" \
  --ssd-path /tmp/ssd-bench

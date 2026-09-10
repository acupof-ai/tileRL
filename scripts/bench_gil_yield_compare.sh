#!/usr/bin/env bash
# Compare decode throughput with and without the GIL yield fix.
set -euo pipefail
cd "$(dirname "$0")/.."
ENGINE=src/tilerl/engine.py
CARD="${CARD:-3}"

run_bench() {
  python3 scripts/bench_batch_decode.py /host/tc27-nvfp4-slice4 \
    --layers 4 --batches 1,8 --fuse --decode-graph \
    --card "$CARD" --device-name "NVIDIA H20" 2>&1 | grep -E '^\s+[0-9]+\s|==='
}

echo "=== Run 1: WITH GIL yield ==="
run_bench

echo ""
echo "=== Patching out sleep(0) ==="
python3 -c "
import re
src = open('$ENGINE').read()
patched = re.sub(r'^(\s+)time\.sleep\(0\)', r'\1# PATCHED: time.sleep(0)', src, flags=re.M)
open('$ENGINE','w').write(patched)
print(f'patched {src.count(\"time.sleep(0)\")} -> {patched.count(\"time.sleep(0)\")} occurrences')
"
find . -name '__pycache__' -path '*/tilerl/*' -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "=== Run 2: WITHOUT GIL yield ==="
run_bench

echo ""
echo "=== Restoring fix ==="
python3 -c "
src = open('$ENGINE').read()
restored = src.replace('# PATCHED: time.sleep(0)', 'time.sleep(0)')
open('$ENGINE','w').write(restored)
print(f'restored: {restored.count(\"time.sleep(0)\")} occurrences')
"
echo "=== Done ==="

#!/usr/bin/env bash
# Item 1: split the 32.9k sparse W2048 trunk graph replay into kernel classes.
# Bucket 2048 only (the 32.9k decision point). Run AFTER the wr sweep, same
# card, production already stopped.
#
#   bash scripts/run_replay_split_v100.sh <outdir>
#
# --cuda-graph-trace=node is load-bearing: without it every kernel inside the
# captured graph collapses into one graph-instantiate/replay entry and the
# class table is meaningless. The parser positive-controls it: node tracing
# expands a 64-layer replay to several hundred kernels per tick; <100/tick
# rc14. --capture-range-end=stop stops the trace at cudaProfilerStop.
set -u

OUT=${1:?$'usage: run_replay_split_v100.sh <outdir>'}
mkdir -p "$OUT"
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda TILELANG_CACHE_DIR=$HOME/.tilelang_cache TMPDIR=$HOME/tmp
export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4
export H2_COLD_BYTES=1073741824 H2_COLD_SSD=$HOME/sparse_cold_128k.bin
export H2_COLD_SSD_BYTES=8589934592 H2_COLD_FORMAT=f16
mkdir -p "$TMPDIR"
PY=$HOME/venv70/bin/python
TICKS=20

if pgrep -f "cli serve" >/dev/null; then
  echo "FATAL a serve is still running — run after the sweep killed it" >&2
  exit 91
fi

echo "===== NSYS CAPTURE b2048 $(date +%T) ====="
nsys profile -t cuda --stats=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o "$OUT/replay_b2048" \
  $PY -u scripts/probe_replay_kernel_split.py --bucket 2048 --nticks "$TICKS" \
  > "$OUT/b2048.nsys.out" 2> "$OUT/b2048.nsys.err"
echo "NSYS EXIT=$?"

# --stats output lands in stderr for nsys 2022.4; concat both for the table.
cat "$OUT/b2048.nsys.out" "$OUT/b2048.nsys.err" > "$OUT/b2048.stats.txt"
ROOF=$(grep -o '"roofline_ms_at_900GBs": [0-9.]*' "$OUT/b2048.nsys.out" | grep -o '[0-9.]*$')
$PY scripts/parse_nsys_kern_sum.py "$OUT/b2048.stats.txt" \
  --ticks "$TICKS" --min-per-tick 100 --roof-ms "${ROOF:-0}" \
  --out "$OUT/replay_split_b2048.json" | tee "$OUT/replay_split_b2048.txt"
echo "REPLAY SPLIT END $(date +%T) dir=$OUT"

# Last window of the pair: bring the card back up as the latest-main service.
# Validated cutover: run_serve_prod.sh. Not yet validated: caller sets
# START_SERVE_CMD to a serve carrying --sparse-min-tokens 8192.
START_SERVE_CMD=${START_SERVE_CMD:-"bash $HOME/run_serve_prod.sh"}
nohup bash -c "$START_SERVE_CMD" > "$OUT/restore_serve.log" 2>&1 < /dev/null &
for _ in $(seq 1 90); do
  H=$(curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null)
  echo "$H" | grep -q '"status":"ok"' && { echo HEALTH_OK; break; }
  sleep 5
done
echo "$H" | head -c 400
echo
curl -s -m 60 -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"reply with the single word ok"}],"temperature":0,"max_tokens":4,"enable_thinking":false}' \
  | python3 -c "import sys,json;print('CHAT200',json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null \
  || echo "CHAT failed — check restore_serve.log"
echo "RESTORE PID=$(pgrep -f 'cli serve' | head -1)"

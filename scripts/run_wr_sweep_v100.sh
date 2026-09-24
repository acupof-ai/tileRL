#!/usr/bin/env bash
# W×R local-window/refresh sweep on V100 sm70, 8 arms, one 27B process each
# (two engines in one process OOM). Serial. Stop production first; the card
# is a test machine and the cutover leaves the latest main as the service,
# so this does NOT restore anything.
#
#   bash scripts/run_wr_sweep_v100.sh <outdir>
#
# Requires: tree synced to the box, prompts at ~/serve805_prompts.jsonl.
set -u

OUT=${1:?$'usage: run_wr_sweep_v100.sh <outdir>'}
mkdir -p "$OUT"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda TILELANG_CACHE_DIR=$HOME/.tilelang_cache TMPDIR=$HOME/tmp
export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4
export H2_COLD_BYTES=1073741824 H2_COLD_SSD=$HOME/sparse_cold_128k.bin
export H2_COLD_SSD_BYTES=8589934592 H2_COLD_FORMAT=f16
mkdir -p "$TMPDIR"
PY=$HOME/venv70/bin/python

SP=$(pgrep -f "cli serve" | head -1)
if [ -n "$SP" ]; then
  LA=$(ps -o ppid= -p "$SP" | tr -d ' ')
  echo "stop production python=$SP launcher=$LA"
  kill -INT "$SP"
  for _ in $(seq 1 30); do sleep 2; kill -0 "$SP" 2>/dev/null || break; done
  kill -0 "$LA" 2>/dev/null && kill "$LA"
  for _ in $(seq 1 60); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "${M:-0}" -lt 1000 ] && break
    sleep 2
  done
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader

for W in 128 1024; do
  for R in 1 8 16 32; do
    TAG=arm_W${W}_R${R}
    echo "===== ARM $TAG $(date +%T) ====="
    $PY -u scripts/probe_wr_sweep_worker.py \
      --window-tokens "$W" --refresh "$R" \
      --prompts "$HOME/serve805_prompts.jsonl" --n-prompts 6 \
      --out-prefix "$OUT/$TAG" > "$OUT/$TAG.out" 2> "$OUT/$TAG.err"
    echo "$TAG EXIT=$?"
  done
done

$PY scripts/wr_sweep_report.py --dir "$OUT" --out "$OUT/wr_sweep.json" | tee "$OUT/summary.txt"
echo "SWEEP END $(date +%T) dir=$OUT; card left without a serve"

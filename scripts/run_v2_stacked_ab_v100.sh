#!/usr/bin/env bash
# PROBE-ONLY #805: stacked-combo A/B speed read on V100, W1024/R32, narrow q.
#
# Everything stacked at once: sparse own-window W1024, refresh R=32, and
# TILERL_DRAFT_TRUE_Q_WIDTH=1 (read at module import, hence exported into the
# worker processes). Two pure FREE runs, one engine per subprocess:
#   A = v2 OFF (lag off)   B = v2 ON (TILERL_SPARSE_V2=async)
# First two serve805 prompts only (~25 min; prefill dominates). Same warm
# caliber as scripts/v2_baseline_24p915.json: ticks [16,end), close tick
# dropped, no prefill. The worker hard-asserts lag_enabled precedes graph
# capture and B==1 every tick.
#
# References: A measured 39.0 in impl's ABBA tonight; B upper-bound est. 44.9.
# If B >= 40 warm tok/s, run the teacher-forced quality follow-up:
#   bash scripts/run_v2_quality_followup_v100.sh <outdir>
#
#   bash scripts/run_v2_stacked_ab_v100.sh <outdir>
set -u

OUT=${1:?$'usage: run_v2_stacked_ab_v100.sh <outdir>'}
mkdir -p "$OUT"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda TILELANG_CACHE_DIR=$HOME/.tilelang_cache TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4
# Narrow draft query; read at `import tilerl.spec`, must be in the worker env.
export TILERL_DRAFT_TRUE_Q_WIDTH=1
PY=$HOME/venv70/bin/python
DRAFT=${DRAFT:-$HOME/mmlu-assets/model_mtp.safetensors}
SSD=${SSD:-$HOME/sparse_cold_128k.bin}
PROMPTS=${PROMPTS:-$HOME/serve805_prompts.jsonl}
export H2_COLD_BYTES=1073741824 H2_COLD_SSD="$SSD" H2_COLD_SSD_BYTES=8589934592
TREE=$(git rev-parse --short=8 HEAD 2>/dev/null || cat .synced_commit)
TREE=${TREE:0:8}

# Stop the production service the same way the other windows do; the restore
# runs only at the very end (after any follow-up; SKIP_SERVE_RESTORE=1 to hold).
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

echo "===== STACKED A/B W1024/R32 narrow-q tree=$TREE $(date +%T) ====="
$PY -u scripts/probe_v2_window.py --ab-free \
  --model qwen38-27b --source "$TILERL_QWEN38_SOURCE" --draft "$DRAFT" \
  --prompts "$PROMPTS" --expect-tree "$TREE" \
  --window-tokens 1024 --refresh 32 --n-prompts 2 \
  --max-new-tokens 512 --min-tokens 20000 --max-tokens 40000 \
  --out-prefix "$OUT/ab" --per-prompt-dir "$OUT/ab_pp" \
  > "$OUT/ab.out" 2> "$OUT/ab.err"
RC=$?
echo "A/B driver EXIT=$RC (0 GO-read ok, 14 instrument; speed itself is not gated)"
cat "$OUT/ab_verdict.json"

if [ "${SKIP_SERVE_RESTORE:-0}" = 1 ]; then
  echo "SKIP_SERVE_RESTORE set; card left stopped (run the quality follow-up next)"
  exit "$RC"
fi
RESTORE_CMD=${START_SERVE_CMD:-"bash $HOME/run_serve_prod.sh"}
nohup bash -c "$RESTORE_CMD" > "$OUT/restore_serve.log" 2>&1 < /dev/null &
for _ in $(seq 1 90); do
  H=$(curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null)
  echo "$H" | grep -q '"status":"ok"' && { echo RESTORE_HEALTH_OK; break; }
  sleep 5
done
echo "$H" | head -c 400; echo
exit "$RC"

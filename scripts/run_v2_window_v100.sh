#!/usr/bin/env bash
# PROBE-ONLY #805: v2 async-refresh device window, two R×W geometries.
#
# Each config runs the 3-subprocess driver (controlA off -> v2 async ->
# controlB off), same 6 serve805 prompts, warm window ticks [16,end) with the
# close tick dropped and no prefill (scripts/v2_baseline_24p915.json caliber).
# The worker hard-asserts sf.lag_enabled is set BEFORE make_sparse_graph
# capture (capture_order_violations -> rc14) and B==1 every graph tick.
#
#   W1024/R32  the sweep's primary proposal; v2 increment est. only 5-10%
#   W128/R8    production window/interval; the settled f6cb7731 config
#
# One engine per subprocess (two 27B in one process OOM on the V100). Serial,
# ~70 min per config (6 prompts, prefill dominated). Stop production first.
#
#   bash scripts/run_v2_window_v100.sh <outdir>
#   START_AT=W1024_R32|W128_R8  resume one config
set -u

OUT=${1:?$'usage: run_v2_window_v100.sh <outdir>'}
mkdir -p "$OUT"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda TILELANG_CACHE_DIR=$HOME/.tilelang_cache TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4
PY=$HOME/venv70/bin/python
DRAFT=${DRAFT:-$HOME/mmlu-assets/model_mtp.safetensors}
SSD=${SSD:-$HOME/sparse_cold_128k.bin}
PROMPTS=${PROMPTS:-$HOME/serve805_prompts.jsonl}
export H2_COLD_BYTES=1073741824 H2_COLD_SSD="$SSD" H2_COLD_SSD_BYTES=8589934592
TREE=$(git rev-parse --short=8 HEAD)

# Free the card the same way the R×W sweep does: SIGINT the server, then its
# launcher; wait for <1 GiB used. Never starts a service — this window owns the
# card until both configs finish.
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

run_cfg() {
  local W=$1 R=$2 TAG=$3
  echo "===== $TAG W=$W R=$R tree=$TREE $(date +%T) ====="
  $PY -u scripts/probe_v2_window.py \
    --model qwen38-27b --source "$TILERL_QWEN38_SOURCE" --draft "$DRAFT" \
    --prompts "$PROMPTS" --expect-tree "$TREE" \
    --window-tokens "$W" --refresh "$R" \
    --max-new-tokens 512 --min-tokens 20000 --max-tokens 40000 \
    --out-prefix "$OUT/$TAG" --per-prompt-dir "$OUT/${TAG}_pp" \
    > "$OUT/$TAG.out" 2> "$OUT/$TAG.err"
  echo "$TAG EXIT=$? rc-def: 0 GO, 1 measured no-go, 14 instrument"
}

# START_AT=<tag> resumes from that config onward; default runs both.
SEQ=(W1024_R32 W128_R8)
started=0
for step in "${SEQ[@]}"; do
  [ "$started" = 0 ] && { [ "$step" = "${START_AT:-W1024_R32}" ] && started=1 || continue; }
  case "$step" in
    W1024_R32) run_cfg 1024 32 "$step" ;;
    W128_R8)   run_cfg 128   8 "$step" ;;
  esac
done

echo "V2 WINDOW END $(date +%T) tree=$TREE dir=$OUT"
echo "verdicts:"
for t in W1024_R32 W128_R8; do
  [ -f "$OUT/${t}_verdict.json" ] && echo "== $t ==" && \
    python3 -c "import json;d=json.load(open('$OUT/${t}_verdict.json'));print(json.dumps({k:d.get(k) for k in ('GO','v2','floor_top1_eq_1_kl_le_eps')},indent=1)[:800])"
done

# Queue rule: hand the card back to the latest-main production service and
# verify /health 200 before exiting. SKIP_SERVE_RESTORE=1 hands the stopped
# card straight to the next queued probe instead.
if [ "${SKIP_SERVE_RESTORE:-0}" = 1 ]; then
  echo "SKIP_SERVE_RESTORE set; card left stopped for the next queue item"
  exit 0
fi
RESTORE_CMD=${START_SERVE_CMD:-"bash $HOME/run_serve_prod.sh"}
nohup bash -c "$RESTORE_CMD" > "$OUT/restore_serve.log" 2>&1 < /dev/null &
for _ in $(seq 1 90); do
  H=$(curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null)
  echo "$H" | grep -q '"status":"ok"' && { echo RESTORE_HEALTH_OK; break; }
  sleep 5
done
echo "$H" | head -c 400; echo

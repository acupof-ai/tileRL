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

# Negative control first: R=1 self-feed with a one-position-shifted anchor must
# fail the committed-output identity. The worker exits 0 only when the red is
# observed; a non-zero exit aborts before any arm burns card time.
echo "===== NEG CONTROL (shifted anchor must diverge) $(date +%T) ====="
$PY -u scripts/probe_wr_sweep_worker.py \
  --window-tokens 128 --refresh 1 \
  --prompts "$HOME/serve805_prompts.jsonl" --n-prompts 1 \
  --neg-anchor-offset 1 --neg-tokens 64 \
  --out-prefix "$OUT/neg_shift" > "$OUT/neg_shift.out" 2> "$OUT/neg_shift.err"
NEGRC=$?
grep -q "NEG-OK" "$OUT/neg_shift.out" || NEGRC=1
if [ "$NEGRC" != "0" ]; then
  echo "FATAL negative control did not go red (rc=$NEGRC); aborting" >&2
  exit 92
fi
echo "NEG CONTROL red as required"

# Shortened 6-arm order: W128 {R1 anchor, R8 free-only speed control, R16,
# R32}, then W1024 {R1 anchor, R32}. 512 tokens × 6 prompts. If W128 R16's
# TF top1 fails the >=0.99 gate, R32 is skipped and we go straight to W1024.
run_arm() {
  local W=$1 R=$2 FREE=${3:-0}
  local TAG=arm_W${W}_R${R}
  echo "===== ARM $TAG $(date +%T) ====="
  local args=(--window-tokens "$W" --refresh "$R"
    --prompts "$HOME/serve805_prompts.jsonl" --n-prompts 6
    --max-new-tokens 512 --out-prefix "$OUT/$TAG")
  if [ "$FREE" = "1" ]; then
    args+=(--free-only)
  elif [ "$R" != "1" ]; then
    # Quality arm: teacher-force this window's R=1 free run.
    args+=(--anchor-dir "$OUT/arm_W${W}_R1_pp")
  fi
  $PY -u scripts/probe_wr_sweep_worker.py "${args[@]}" \
    > "$OUT/$TAG.out" 2> "$OUT/$TAG.err"
  echo "$TAG EXIT=$?"
}

run_arm 128 1
run_arm 128 8 1
run_arm 128 16
# Skip W128 R32 if R16's mean TF top1 agreement is below 0.99.
if $PY - "$OUT" <<'PY'
import glob, json, sys
ag = [json.load(open(f))["top1_agreement_vs_anchor"]
      for f in sorted(glob.glob(f"{sys.argv[1]}/arm_W128_R16_tf/tf_*.json"))]
mean = sum(ag) / len(ag) if ag else 0.0
print(f"W128 R16 mean top1={mean:.4f} -> {'RUN R32' if mean >= 0.99 else 'SKIP R32'}")
sys.exit(0 if mean >= 0.99 else 3)
PY
then
  run_arm 128 32
else
  echo "W128 R16 failed top1 gate; skipping W128 R32"
fi
run_arm 1024 1
run_arm 1024 32

$PY scripts/wr_sweep_report.py --dir "$OUT" --out "$OUT/wr_sweep.json" | tee "$OUT/summary.txt"

echo "SWEEP END $(date +%T) dir=$OUT"
# When chained with the replay-split window (item 2 then item 1), restore once,
# AFTER both: SKIP_SERVE_RESTORE=1 hands the stopped card straight to the next
# window.
if [ "${SKIP_SERVE_RESTORE:-0}" = "1" ]; then
  echo "SKIP_SERVE_RESTORE set; card left stopped for the next window"
  exit 0
fi
# Do not leave the test card idle: start the latest-main service. After a
# validated fixmisc cutover run_serve_prod.sh IS the cutover config; if the
# cutover is not validated yet, the handover note sets START_SERVE_CMD to an
# explicit serve with --sparse-min-tokens 8192.
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

#!/usr/bin/env bash
# PROBE-ONLY #805: teacher-forced QUALITY follow-up to the stacked A/B read.
#
# Run only when the A/B read's B (v2 on) hit >= 40 warm tok/s. Same geometry
# (W1024/R32) and narrow draft query as the speed read. Runs the full
# 3-subprocess driver (controlA off -> v2 async -> controlB off), which forces
# v2/controlB onto controlA's free-run token stream and gates the floor
# (controlB vs controlA top1=1.0, KL<=1e-4, coverage) and the binding quality
# (v2 tf mean top1 >= 0.99, coverage; KL/margins report-only).
#
#   bash scripts/run_v2_quality_followup_v100.sh <outdir>
#
# First two serve805 prompts, same warm caliber as the speed read. Restores the
# production service and verifies /health at the end (SKIP_SERVE_RESTORE=1 to
# hand the stopped card on).
set -u

OUT=${1:?$'usage: run_v2_quality_followup_v100.sh <outdir>'}
mkdir -p "$OUT"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda TILELANG_CACHE_DIR=$HOME/.tilelang_cache TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4
export TILERL_DRAFT_TRUE_Q_WIDTH=1
PY=$HOME/venv70/bin/python
DRAFT=${DRAFT:-$HOME/mmlu-assets/model_mtp.safetensors}
SSD=${SSD:-$HOME/sparse_cold_128k.bin}
PROMPTS=${PROMPTS:-$HOME/serve805_prompts.jsonl}
export H2_COLD_BYTES=1073741824 H2_COLD_SSD="$SSD" H2_COLD_SSD_BYTES=8589934592
TREE=$(git rev-parse --short=8 HEAD 2>/dev/null || cat .synced_commit)
TREE=${TREE:0:8}

# Free the card if a prior window left a service up.
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

echo "===== QUALITY FOLLOW-UP W1024/R32 narrow-q tree=$TREE $(date +%T) ====="
$PY -u scripts/probe_v2_window.py --quality-only \
  --model qwen38-27b --source "$TILERL_QWEN38_SOURCE" --draft "$DRAFT" \
  --prompts "$PROMPTS" --expect-tree "$TREE" \
  --window-tokens 1024 --refresh 32 --n-prompts 2 \
  --max-new-tokens 512 --min-tokens 20000 --max-tokens 40000 \
  --out-prefix "$OUT/qf" --per-prompt-dir "$OUT/qf_pp" \
  > "$OUT/qf.out" 2> "$OUT/qf.err"
RC=$?
echo "quality driver EXIT=$RC (0 GO incl. quality, 1 measured no-go, 14 instrument)"
python3 -c "import json;d=json.load(open('$OUT/qf_verdict.json'));g=d['gates'];print(json.dumps({'floor':d['floor_top1_eq_1_kl_le_eps'],'tf_mean_top1':g['tf_mean_top1_agreement'],'tf_min_top1':g['tf_min_top1_agreement'],'tf_kl_ab':g['tf_mean_kl_ab'],'tf_cov':g['tf_coverage_ok'],'GO':d['GO']},indent=2))"

if [ "${SKIP_SERVE_RESTORE:-0}" = 1 ]; then
  echo "SKIP_SERVE_RESTORE set; card left stopped"
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

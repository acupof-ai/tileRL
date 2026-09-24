#!/usr/bin/env bash
# PROBE-ONLY: ThinkingCap-Qwen3.8-27B-NVFP4 on-card validation on the V100.
#
# Same window, same 6 serve805 prompts, same streaming client, two services:
# the OLD model first (production launcher verbatim), then ThinkingCap from a
# launcher that differs from the production one in EXACTLY two lines
# (TILERL_QWEN38_SOURCE and --draft). Relative acceptance gate:
# ThinkingCap accept >= 0.8 x old, in-thinking/post-thinking split reported.
#
# Cadence: if #821 is in the tree, one service per model; otherwise each prompt
# gets a FRESH service (avoids the cross-request refresh-phase carry-over).
# FORCE_FRESH=1 forces restart-per-prompt regardless.
#
#   bash scripts/run_thinkingcap_validate_v100.sh <outdir>
set -u

OUT=${1:?$'usage: run_thinkingcap_validate_v100.sh <outdir>'}
mkdir -p "$OUT"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD/packages/tilerl-kernels/src
export TILERL_TARGET=cuda TILELANG_CACHE_DIR=$HOME/.tilelang_cache TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
PY=$HOME/venv70/bin/python
PROMPTS=${PROMPTS:-$HOME/serve805_prompts.jsonl}
URL=http://127.0.0.1:8000
PORT=8000
PROD_LAUNCHER=$HOME/run_serve_82e3_min0.sh
TC_SRC=$HOME/models/ThinkingCap-Qwen3.8-27B-NVFP4
TC_DRAFT=$TC_SRC/model-base-aux.safetensors
TC_LAUNCHER=$OUT/run_serve_thinkingcap_min0.sh

[ -f "$PROD_LAUNCHER" ] || { echo "missing $PROD_LAUNCHER" >&2; exit 2; }
[ -d "$TC_SRC" ] || { echo "missing model dir $TC_SRC" >&2; exit 2; }
[ -f "$TC_DRAFT" ] || { echo "missing draft $TC_DRAFT" >&2; exit 2; }

# ---- the 2-line launcher, self-checked ------------------------------------
cp "$PROD_LAUNCHER" "$TC_LAUNCHER"
python3 - "$TC_LAUNCHER" "$TC_SRC" "$TC_DRAFT" <<'PYEOF'
import re, sys
p, src, draft = sys.argv[1:4]
s = open(p).read()
s2, n1 = re.subn(r'(export TILERL_QWEN38_SOURCE=).*', rf'\g<1>{src}', s)
s3, n2 = re.subn(r'(--draft\s+)(?:"[^"]*"|\S+)', rf'\g<1>"{draft}"', s2)
assert n1 == 1, f"TILERL_QWEN38_SOURCE line matched {n1} times, need exactly 1"
assert n2 == 1, f"--draft matched {n2} times, need exactly 1"
open(p, "w").write(s3)
PYEOF
CHANGED=$(diff "$PROD_LAUNCHER" "$TC_LAUNCHER" | grep -c '^< ')
echo "launcher diff (changed source lines: $CHANGED, need 2):"
diff "$PROD_LAUNCHER" "$TC_LAUNCHER" || true
[ "$CHANGED" = 2 ] || { echo "FATAL launcher differs in $CHANGED lines, need exactly 2" >&2; exit 2; }

# ---- cadence detection: #821 reset in this tree => one service per model ----
if [ "${FORCE_FRESH:-0}" = 1 ]; then
  FRESH=1
elif git grep -q "refresh phase" -- src/tilerl/engine.py 2>/dev/null \
     || git log --oneline -300 | grep -q "refresh phase at zero"; then
  FRESH=0
else
  FRESH=1
fi
echo "cadence #821 present => fresh-per-prompt FRESH=$FRESH (1=restart each prompt)"

# ---- process management ----------------------------------------------------
# The production launcher is a flock'd supervisor that restarts its python and
# traps TERM to kill the child. Start it in its own process group and kill the
# WHOLE group, or a restart-loop child survives across the model swap.
SV_PGID=
kill_existing() {
  # Free port 8000 and the launcher's flock before starting, or the new
  # supervisor exits "already running" and health answers from the old model.
  local sp la
  sp=$(pgrep -f "cli serve" | head -1)
  if [ -n "$sp" ]; then
    la=$(ps -o ppid= -p "$sp" | tr -d ' ')
    echo "preflight: stop existing server python=$sp launcher=$la"
    kill -INT "$sp" 2>/dev/null
    sleep 3
    [ -n "$la" ] && kill "$la" 2>/dev/null
  fi
  pkill -f "run_serve_82e3_min0|run_serve_thinkingcap_min0" 2>/dev/null
  for _ in $(seq 1 40); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "${M:-0}" -lt 1000 ] && break
    sleep 2
  done
}

start_server() {
  local launcher=$1
  kill_existing
  setsid bash "$launcher" < /dev/null >> "$OUT/server_${2}.log" 2>&1 &
  SV_PGID=$!
  for _ in $(seq 1 300); do
    H=$(curl -s -m 3 "$URL/health" 2>/dev/null) && echo "$H" | grep -q '"status":"ok"' && {
      echo "server $2 healthy"; return 0; }
    kill -0 -- "-$SV_PGID" 2>/dev/null || { echo "server $2 exited early"; \
      tail -30 "$OUT/server_${2}.log" >&2; return 1; }
    sleep 2
  done
  echo "server $2 never healthy" >&2; return 1
}
stop_server() {
  [ -z "$SV_PGID" ] && return 0
  kill -TERM -- "-$SV_PGID" 2>/dev/null
  sleep 8
  kill -KILL -- "-$SV_PGID" 2>/dev/null
  for _ in $(seq 1 40); do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ "${M:-0}" -lt 1000 ] && break
    sleep 2
  done
  SV_PGID=
}
trap 'stop_server; exit 143' TERM INT

restore_and_exit() {
  local rc=$1
  stop_server
  if [ "${SKIP_SERVE_RESTORE:-0}" = 1 ]; then
    echo "SKIP_SERVE_RESTORE set; card left stopped"; exit "$rc"
  fi
  local RESTORE_CMD=${START_SERVE_CMD:-"bash $HOME/run_serve_prod.sh"}
  nohup bash -c "$RESTORE_CMD" > "$OUT/restore_serve.log" 2>&1 < /dev/null &
  local H=
  for _ in $(seq 1 90); do
    H=$(curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null)
    echo "$H" | grep -q '"status":"ok"' && { echo RESTORE_HEALTH_OK; break; }
    sleep 5
  done
  echo "$H" | head -c 400; echo
  exit "$rc"
}

run_client() { # tag prompt_idx_or_all
  local tag=$1 idx=$2
  if [ "$idx" = all ]; then
    $PY -u scripts/tc_validate_client.py --url "$URL" --prompts "$PROMPTS" \
      --n-prompts 6 --max-new 512 --tag "$tag" --out-prefix "$OUT/$tag"
  else
    $PY -u scripts/tc_validate_client.py --url "$URL" --prompts "$PROMPTS" \
      --only-prompt "$idx" --max-new 512 --tag "$tag" \
      --out-prefix "$OUT/${tag}"
  fi
}

run_model() { # tag launcher
  local tag=$1 launcher=$2
  if [ "$FRESH" = 0 ]; then
    start_server "$launcher" "$tag" || return 1
    # one short chat each mode BEFORE the 32k set: default, then thinking off
    $PY -u scripts/tc_validate_client.py --url "$URL" --short-chat \
      --tag "${tag}_short_default" --out-prefix "$OUT/${tag}_short_default"
    $PY -u scripts/tc_validate_client.py --url "$URL" --short-chat --thinking-off \
      --tag "${tag}_short_off" --out-prefix "$OUT/${tag}_short_off"
    run_client "$tag" all
    stop_server
  else
    for i in 0 1 2 3 4 5; do
      start_server "$launcher" "${tag}_p$i" || return 1
      [ "$i" = 0 ] && {
        $PY -u scripts/tc_validate_client.py --url "$URL" --short-chat \
          --tag "${tag}_short_default" --out-prefix "$OUT/${tag}_short_default"
        $PY -u scripts/tc_validate_client.py --url "$URL" --short-chat --thinking-off \
          --tag "${tag}_short_off" --out-prefix "$OUT/${tag}_short_off"; }
      run_client "$tag" "$i"
      stop_server
    done
    $PY - "$OUT" "$tag" <<'PYEOF'
import json, glob, os, sys
out, tag = sys.argv[1:3]
rows = []
for f in sorted(glob.glob(os.path.join(out, f"{tag}_[0-9]*.json"))):
    rows += json.load(open(f))["prompts"]
d = sum(r["spec_drafted"] for r in rows)
a = sum(r["spec_accepted"] for r in rows)
rep = {"tag": tag, "prompts": rows,
       "aggregate_accept_rate": round(a/d, 4) if d else None,
       "aggregate_spec_accepted": a, "aggregate_spec_drafted": d,
       "aggregate_eff_tok_s": round(sum(r["completion_tokens"] or 0 for r in rows)
                                    / sum(r["wall_s"] for r in rows), 3)}
json.dump(rep, open(os.path.join(out, f"{tag}.json"), "w"), indent=2, ensure_ascii=False)
print("merged", tag, rep["aggregate_accept_rate"])
PYEOF
  fi
}

# OLD first, then ThinkingCap
run_model old "$PROD_LAUNCHER" || restore_and_exit 14
run_model thinkingcap "$TC_LAUNCHER" || restore_and_exit 14

$PY scripts/tc_validate_client.py --compare "$OUT/old.json" "$OUT/thinkingcap.json" \
  | tee "$OUT/acceptance_gate.json"
GATE_RC=${PIPESTATUS[0]}
echo "VALIDATION END $(date +%T) dir=$OUT fresh_per_prompt=$FRESH"
restore_and_exit "$GATE_RC"

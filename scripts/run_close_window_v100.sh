#!/usr/bin/env bash
# One-key orchestration for the final V100 close-tail window.
#
# Why this exists: the 2026-09-19/20 windows each lost time to the same manual
# steps -- piloting on a tiny model, forgetting the cold-tier fill gate, leaving
# liveness polling on (it sends real chats every 60 s and poisons the decode
# window), reading one operand in mixed units, and hand-editing env between arms.
# This script does those steps in one order, checks the health body it needs,
# and writes every artifact to a vendored directory so a number is attributable.
#
# It orchestrates; it does not measure. The probes it calls own their reporting:
#   scripts/probe_headroom_coldtail.py  arm / compare / reclaim-sample  (#620/#749)
#   scripts/probe_sse_overload.py       --timing        (cancel immediacy)
# and the serve itself is the shipped supervisor:
#   scripts/serve_hybrid_v100.sh
#
#   scripts/run_close_window_v100.sh --repo ~/tilerl-v100-sse --out ~/closewin
#   scripts/run_close_window_v100.sh --list
#   scripts/run_close_window_v100.sh --arm bg1        # one arm
#   scripts/run_close_window_v100.sh --all            # every arm, then restore
#   scripts/run_close_window_v100.sh --restore-only   # official serve back, no flags
#
# NOT run from CI or a GPU-less host: every arm boots a 27B serve.
set -u

REPO=${SERVE_REPO:-$HOME/tilerl-v100-sse}
PORT=${SERVE_PORT:-8000}
PYTHON=${SERVE_PYTHON:-$HOME/venv70/bin/python}
ROOT=${SERVE_ROOT:-$HOME}
OUT=${OUT:-$ROOT/closewin}
LOG=$ROOT/servehybridsse.log
COLD_SSD=${SERVE_COLD_SSD:-$ROOT/sparse_cold_128k.bin}
HEALTH_URL="http://127.0.0.1:$PORT"
WINDOW_TOKENS=${WINDOW_TOKENS:-2048}
EXPECT_BLOCKS=${EXPECT_BLOCKS:-2213}
EXPECT_MODEL=${EXPECT_MODEL:-qwen38-27b}
PROMPT_TOKENS=${PROMPT_TOKENS:-32000}
WARM_REPS=${WARM_REPS:-3}
RECLAIM_SAMPLES=${RECLAIM_SAMPLES:-120}
RECLAIM_INTERVAL_S=${RECLAIM_INTERVAL_S:-20}

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

# ---------------------------------------------------------------- arm table
# Each arm is the env delta over the shared serve command. Order is deliberate:
# cheapest change first, each arm adding one flag so a delta is attributable.
# The lock-split arm (#746) is a placeholder until that PR merges -- an unmerged
# flag would boot a serve that silently ignores it and report a no-op as a result.
#
# bg1 vs bg2 is the depth question, not a second flag: TILERL_CLOSE_BG_PUBLISH
# alone leaves build.py to derive (num_slots+1)*ceil(max_ctx/16) from the shape,
# while bg1 pins the queue at its own default so the two are separable. bg3 pins
# the 8192 the 2026-09-20 window measured clean.
ARM_NAMES=(baseline batch bg1 bg2 bg3 locksplit)
arm_env() {
  case "$1" in
    baseline)  echo "" ;;
    batch)     echo "TILERL_CLOSE_BATCH_D2H=1" ;;
    bg1)       echo "TILERL_CLOSE_BATCH_D2H=1 TILERL_CLOSE_BG_PUBLISH=1 TILERL_CLOSE_BG_DEPTH=512" ;;
    bg2)       echo "TILERL_CLOSE_BATCH_D2H=1 TILERL_CLOSE_BG_PUBLISH=1" ;;
    bg3)       echo "TILERL_CLOSE_BATCH_D2H=1 TILERL_CLOSE_BG_PUBLISH=1 TILERL_CLOSE_BG_DEPTH=8192" ;;
    locksplit) echo "PENDING_746" ;;
    *) return 2 ;;
  esac
}

# Instrumentation every arm runs with, set once here rather than per arm.
# LIVENESS_POLL_S=999999 is the load-bearing one: the supervisor's liveness probe
# sends a REAL chat every 60 s, which lands in the decode window being measured.
instrument_env() {
  echo "LIVENESS_POLL_S=999999 TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0 TILERL_CLOSE_BUSYIDLE=1"
}

ARMS=()
ALL=0
RESTORE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --port) PORT=$2; HEALTH_URL="http://127.0.0.1:$PORT"; shift 2 ;;
    --arm) ARMS+=("$2"); shift 2 ;;
    --all) ALL=1; shift ;;
    --restore-only) RESTORE_ONLY=1; shift ;;
    --list) printf '%s\n' "${ARM_NAMES[@]}"; exit 0 ;;
    --arms-env) for _a in "${ARM_NAMES[@]}"; do printf '%s|%s\n' "$_a" "$(arm_env "$_a")"; done; exit 0 ;;
    --instrument-env) printf '%s\n' "$(instrument_env)"; exit 0 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[closewin] $*"; }

# ---------------------------------------------------------------- health gate
# A serve that answers /health is not necessarily the serve this window is about:
# a leftover tiny-model process, a different pool size, or an eager fallback all
# answer 200. Assert the body the arms are compared under.
health_gate() {
  "$PYTHON" - "$HEALTH_URL" "$EXPECT_MODEL" "$EXPECT_BLOCKS" <<'PY'
import json, sys, urllib.request
url, want_model, want_blocks = sys.argv[1], sys.argv[2], int(sys.argv[3])
bad = []
try:
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        body = json.loads(r.read())
except Exception as exc:
    print(f"health unreachable: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
st = body.get("stats") or {}
if body.get("model") != want_model:
    bad.append(f"model={body.get('model')!r} want {want_model!r}")
if st.get("blocks_total") != want_blocks:
    bad.append(f"blocks_total={st.get('blocks_total')!r} want {want_blocks} "
               f"(a different pool is a different experiment)")
# decode_graph rides in stats because a failed capture flips it off at runtime
# and the serve keeps answering; a window arm must not silently run eager.
if st.get("decode_graph") is False:
    bad.append("decode_graph off at runtime (capture failed?) -- the arms assume "
               "the captured decode tick")
print("health gate OK" if not bad else "health gate FAILED: " + "; ".join(bad))
raise SystemExit(1 if bad else 0)
PY
}

wait_ready() {
  for _ in $(seq 1 600); do
    if curl -sf -m 3 -o /dev/null "$HEALTH_URL/health"; then return 0; fi
    sleep 2
  done
  return 1
}

stop_serve() {
  pkill -TERM -f "serve_hybrid_v100.sh" 2>/dev/null
  pkill -TERM -f "tilerl.cli serve" 2>/dev/null
  pkill -TERM -f "serve_liveness.py" 2>/dev/null
  sleep 5
  # Zombies read as alive to kill -0/pgrep; read the state instead.
  for _ in $(seq 1 30); do
    if ! pgrep -f "tilerl.cli serve" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  pgrep -f "tilerl.cli serve" >/dev/null 2>&1 && {
    log "WARN: serve still present after TERM; check ps -o stat="; return 1; }
  return 0
}

# ------------------------------------------------------------------- one arm
run_arm() {
  local name=$1
  local env_delta; env_delta=$(arm_env "$name") || { log "unknown arm $name"; return 2; }
  if [ "$env_delta" = "PENDING_746" ]; then
    log "arm $name: SKIPPED -- #746 (lock split) is not merged; a flag that does"
    log "  nothing would report a no-op as a measured result. Re-run after it lands."
    return 0
  fi
  local dir=$OUT/$name
  mkdir -p "$dir"
  log "=== arm $name: $env_delta"
  stop_serve || return 1

  # Every artifact from this arm lands in $dir; nothing writes outside it.
  local reclaim_pid=
  local env_line; env_line="$(instrument_env) ${env_delta}"
  # shellcheck disable=SC2086  # deliberate: the helpers emit VAR=val words to split
  ( export SERVE_REPO=$REPO SERVE_ROOT=$ROOT SERVE_PORT=$PORT
    setsid nohup env $env_line scripts/serve_hybrid_v100.sh >/dev/null 2>&1 & )
  wait_ready || { log "arm $name: serve never became ready"; return 1; }
  health_gate || { log "arm $name: health gate failed"; stop_serve; return 1; }

  # reclaim-sample is passive and must span the publish refs' release, so it
  # starts before the fill and outlives it; the spill is retained until it ends.
  if [ -f "$COLD_SSD" ]; then
    "$PYTHON" scripts/probe_headroom_coldtail.py reclaim-sample \
      --spill-path "$COLD_SSD" --out "$dir/reclaim.json" \
      --samples "$RECLAIM_SAMPLES" --interval-s "$RECLAIM_INTERVAL_S" \
      >"$dir/reclaim.log" 2>&1 &
    reclaim_pid=$!
  fi

  "$PYTHON" scripts/probe_headroom_coldtail.py arm \
    --url "$HEALTH_URL" --headroom 0 --log "$LOG" --out "$dir/arm.json" \
    --prompt-tokens "$PROMPT_TOKENS" --warm-reps "$WARM_REPS" \
    --expect-window "$WINDOW_TOKENS" >"$dir/arm.log" 2>&1
  local rc=$?
  log "arm $name: probe rc=$rc"

  # The probe's p50 is the WIDE set (dec>0), which keeps captured-graph ticks and
  # the long close-tail ticks; the sweep arms are quoted on
  # dec==1 & sparse==1 & model>0 & sample>0 & path!=graph with the tail listed
  # apart. Re-filter the same log so the two can be tabled together -- or not
  # tabled at all, when this log cannot support the standard set.
  "$PYTHON" scripts/steady_filter.py --log "$LOG" --out "$dir/steady.json" \
    >"$dir/steady.log" 2>&1
  log "arm $name: steady-filter rc=$? (see $dir/steady.json)"

  # Each smoke is recorded, and a failed one fails the arm: a window that reports
  # "arm done" over a broken follower or a wedged cancel is worse than one that
  # reports nothing. The probe's own rc is kept as the primary verdict.
  local rc_f=0 rc_c=0
  "$PYTHON" - "$HEALTH_URL" "$dir" "$PROMPT_TOKENS" >"$dir/follower.log" 2>&1 <<'PY'
import json, sys, urllib.request
url, out, ptok = sys.argv[1], sys.argv[2], int(sys.argv[3])
def health():
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        return json.loads(r.read())["stats"]
def chat(prompt, gen, timeout=1800):
    body = {"model": "qwen38-27b", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": gen, "enable_thinking": False}
    req = urllib.request.Request(url + "/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]
head = " ".join(["closewindow"] * (ptok // 2)) + " remember this head"
h0 = health()
a = chat(head, 24)
h1 = health()
b = chat(head, 24)          # identical prompt: must hit, must match a
h2 = health()
rec = {
    "follower_a_len": len(a), "follower_b_len": len(b),
    "tokens_identical": a == b,
    "prefix_hits_delta": h2.get("prefix_hits", 0) - h1.get("prefix_hits", 0),
    "prefix_hit_tokens_delta": (h2.get("prefix_hit_tokens", 0)
                                - h1.get("prefix_hit_tokens", 0)),
    "kv_cold_drops_delta": h2.get("kv_cold_drops", 0) - h0.get("kv_cold_drops", 0),
    "blocks_within_total": bool(h2.get("blocks_used", 0) <= h2.get("blocks_total", 1)),
}
# Three outcomes with three meanings, and three DISTINCT exit codes so the
# shell's rc says which one happened without reading the JSON:
#   0  OK
#   3  MISMATCH       -- the store answered, with the wrong tokens: a correctness
#                        bug, the arm must not be reported as good.
#   4  NO-PREFIX-HIT  -- same tokens, no hit: the store did not serve. Worth a
#                        result, not a correctness failure.
#   5  BLOCKS         -- a leak, independent of both.
rec["verdict"] = ("MISMATCH" if not rec["tokens_identical"]
                  else "NO-PREFIX-HIT" if rec["prefix_hits_delta"] <= 0
                  else "OK")
with open(out + "/follower.json", "w") as fh:
    json.dump(rec, fh, indent=1)
print(json.dumps(rec))
if not rec["blocks_within_total"]:
    raise SystemExit(5)
raise SystemExit({"OK": 0, "MISMATCH": 3, "NO-PREFIX-HIT": 4}[rec["verdict"]])
PY
  rc_f=$?
  log "arm $name: follower rc=$rc_f (verdict in $dir/follower.json)"

  # Cancel immediacy: real disconnects, asserted against the engine's own
  # release, not a wall clock (the 2026-09-15 lesson).
  "$PYTHON" scripts/probe_sse_overload.py --timing --sse-reps 6 --nonstream-reps 3 \
    --health-log "$dir/health_timing.log" >"$dir/cancel.log" 2>&1
  rc_c=$?
  log "arm $name: cancel rc=$rc_c"

  if [ -n "$reclaim_pid" ]; then
    wait "$reclaim_pid" 2>/dev/null
    log "arm $name: reclaim rc=$?"
  fi
  stop_serve || true
  # Exit codes, one meaning each, so a wrapper can branch without parsing logs:
  #   probe's own rc (13 = fail-closed on reps, ...)   as-is
  #   3 / 4 / 5   follower MISMATCH / NO-PREFIX-HIT / block leak
  #   6           cancel smoke failed
  [ "$rc" != 0 ] && return "$rc"
  [ "$rc_f" != 0 ] && return "$rc_f"
  [ "$rc_c" != 0 ] && return 6
  return 0
}

# ------------------------------------------------------------------ restore
# Back to the shipped serve: every experimental flag off, liveness back to 60 s.
restore() {
  log "=== restore: official serve, no experimental flags"
  stop_serve || return 1
  # LIVENESS back to its shipped 60 s, and NONE of the instrumentation vars: the
  # restore target is the serve the tree would boot with no window running.
  ( export SERVE_REPO=$REPO SERVE_ROOT=$ROOT SERVE_PORT=$PORT
    export LIVENESS_POLL_S=60
    setsid nohup scripts/serve_hybrid_v100.sh >/dev/null 2>&1 & )
  wait_ready || { log "restore: serve never became ready"; return 1; }
  health_gate || log "restore: health gate differs from the expected baseline (check by hand)"
  "$PYTHON" - "$HEALTH_URL" <<'PY'
import json, sys, urllib.request
url = sys.argv[1]
body = {"model": "qwen38-27b", "messages": [{"role": "user", "content": "say ok"}],
        "temperature": 0.0, "max_tokens": 4, "enable_thinking": False}
req = urllib.request.Request(url + "/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    out = json.loads(r.read())["choices"][0]["message"]["content"]
print("restore chat smoke:", repr(out[:40]))
PY
}

# --------------------------------------------------------------- spill cleanup
# Deleting the spill is irreversible and it is what a re-run needs, so it is
# separate, confirmed, and only ever after the serve is confirmed down.
clean_spill() {
  log "spill cleanup requested. Checking the serve is really down first."
  pgrep -f "tilerl.cli serve" >/dev/null 2>&1 && {
    log "REFUSING: a serve is still running; stop it before deleting the spill"; return 1; }
  for f in "$COLD_SSD" "$COLD_SSD.prefix.bin" "$ROOT/$EXPECT_MODEL.prefix.bin"; do
    [ -e "$f" ] || continue
    printf 'delete %s (%s)? [y/N] ' "$f" "$(du -h "$f" | cut -f1)"
    read -r ans
    [ "$ans" = "y" ] && rm -f "$f" && log "deleted $f" || log "kept $f"
  done
}

# ------------------------------------------------------------------------ main
mkdir -p "$OUT"
log "repo=$REPO out=$OUT port=$PORT blocks=$EXPECT_BLOCKS"

if [ "$RESTORE_ONLY" = 1 ]; then restore; exit $?; fi
if [ "$ALL" = 1 ]; then ARMS=("${ARM_NAMES[@]}"); fi
if [ ${#ARMS[@]} -eq 0 ]; then usage; fi

fail=0
for a in "${ARMS[@]}"; do
  run_arm "$a" || fail=1
done
if [ "$fail" = 0 ]; then
  cmp=$(printf '%s ' "${ARMS[@]}")
  log "arm summaries ready; compare with:"
  log "  $PYTHON scripts/probe_headroom_coldtail.py compare --arms $(for a in "${ARMS[@]}"; do printf '%s=%s/%s/arm.json ' "$a" "$OUT" "$a"; done)"
  log "(the compare subcommand takes name=path pairs; input: $cmp)"
fi

printf '\nRestore the official serve now? [y/N] '
read -r ans
[ "$ans" = "y" ] && restore
printf '\nDelete the (regenerable) spill files? [y/N] '
read -r ans
[ "$ans" = "y" ] && clean_spill

log "done. artifacts under $OUT; recovery: --restore-only"
exit "$fail"

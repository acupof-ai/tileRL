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
#   scripts/probe_headroom_coldtail.py  arm / compare  (#620/#749)
#   scripts/probe_sse_overload.py       --timing        (cancel immediacy)
# and the serve itself is the shipped supervisor:
#   scripts/serve_hybrid_v100.sh
#
#   scripts/run_close_window_v100.sh --repo ~/tilerl-v100-sse --out ~/closewin
#   scripts/run_close_window_v100.sh --list
#   scripts/run_close_window_v100.sh --arm baseline   # one arm
#   scripts/run_close_window_v100.sh --all            # every arm, then restore
#   scripts/run_close_window_v100.sh --restore-only   # official serve back, no flags
#   scripts/run_close_window_v100.sh --clean-spill-only  # delete the regenerable spill, no serve
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

# The header comment block, by RULE not by line number: a fixed `sed -n '2,26p'`
# silently printed the wrong lines (and lost the examples) the first time a header
# line was added. Every leading comment line until the first line of code.
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0; }

# ---------------------------------------------------------------- arm table
# Env delta each arm applies over the shared serve command. `baseline` is the
# reference the measured arms are delta against; it is the only one left since the
# close/batch/bg machinery and its arms were deleted (#784, #787).
ARM_NAMES=(baseline)
arm_env() {
  case "$1" in
    baseline)  echo "" ;;
    *) return 2 ;;
  esac
}

# Instrumentation every arm runs with, set once here rather than per arm.
# LIVENESS_POLL_S=999999 is the load-bearing one: the supervisor's liveness probe
# sends a REAL chat every 60 s, which lands in the decode window being measured.
instrument_env() {
  # TILERL_DRAFT_ATTN_WINDOW_TOKENS is here, not in an arm, because it is a
  # MEASUREMENT setting, not a treatment: every arm is compared at the same read
  # window, and the probe asserts it (--expect-window). Omitting it left the
  # loader default W=0 against the probe's 2048 and every arm exited rc13 before
  # producing a number -- a whole window lost to a missing env var.
  #
  # This injects the window for the WINDOW only: it is set in the arm's serve
  # environment here, and DRAFT_ATTN_WINDOW_TOKENS_DEFAULT stays 0, so the
  # shipped serve is untouched by this harness.
  echo "LIVENESS_POLL_S=999999 TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0 TILERL_DRAFT_ATTN_WINDOW_TOKENS=$WINDOW_TOKENS"
}

# The shared-prefix spill is a SET of siblings, not one file. kv_tiers
# `_shared_bucket_path` sends the cold layout (sig = 3 fields) to the plain
# `.prefix.bin` and every OTHER recognized layout to
# `<base>.prefix.w<field-count>.bin` -- w is followed by the signature's FIELD
# COUNT, not a length. Today exactly one exists: the warm spec page, whose sig is
# (bounds,dk,dv,k,v) = 5 fields, so `.prefix.w5.bin`. (_MAX_SHARED_BUCKETS=4 caps
# the bucket count INCLUDING the cold one, so it is not "up to 4 warm files".)
#
# Globbing the stem covers a layout added later without this list learning its
# name, and the stem keeps the match scoped to THIS arm's spill: a sibling under a
# different stem (another arm's, or the model-named one) can never match. Printed
# in the order they are offered for deletion; one per line, so a caller can read
# the list without executing the prompt (`--spill-files` is how the gate does it).
spill_files() {
  local stem="${COLD_SSD%.bin}"
  printf '%s\n' "$COLD_SSD" "$stem.prefix.bin" "$stem".prefix.w*.bin \
                "$ROOT/$EXPECT_MODEL.prefix.bin"
}

ARMS=()
ALL=0
RESTORE_ONLY=0
CLEAN_SPILL_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --port) PORT=$2; HEALTH_URL="http://127.0.0.1:$PORT"; shift 2 ;;
    --arm) ARMS+=("$2"); shift 2 ;;
    --all) ALL=1; shift ;;
    --restore-only) RESTORE_ONLY=1; shift ;;
    --clean-spill-only) CLEAN_SPILL_ONLY=1; shift ;;
    --list) printf '%s\n' "${ARM_NAMES[@]}"; exit 0 ;;
    --arms-env) for _a in "${ARM_NAMES[@]}"; do printf '%s|%s\n' "$_a" "$(arm_env "$_a")"; done; exit 0 ;;
    --instrument-env) printf '%s\n' "$(instrument_env)"; exit 0 ;;
    --spill-files) spill_files; exit 0 ;;
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
  local dir=$OUT/$name
  mkdir -p "$dir"
  log "=== arm $name: $env_delta"
  stop_serve || return 1

  # One log per arm. The supervisor only truncates a log that is ALREADY over
  # 32 MiB at boot (serve_hybrid_v100.sh LOG_CAP), and the probe/steady filter
  # read from offset 0, so a shared fixed path made every arm's steady.json a
  # statistic over all previous arms' ticks as well. SERVE_LOG is the
  # supervisor's own override.
  local arm_log=$dir/serve.log
  : > "$arm_log"

  # One FUSE file per arm, for the same reason and with a worse failure mode.
  # serve_hybrid_v100.sh appends a timestamp on every non-zero-exit restart and
  # stays down (exit 2) once RESTART_FUSE_MAX land inside RESTART_FUSE_WINDOW_S.
  # That state is $ROOT/.servehybridsse.fuse and $ROOT is shared by every arm, so
  # without this an arm that crash-loops >=5 times inside the window leaves a
  # tripped fuse behind and the NEXT arm's serve exits 2 without ever starting
  # python -- which this harness would report as that arm's result (the health
  # gate only says "never became ready"). A number from an arm that never ran is
  # the exact failure this window exists to avoid.
  #
  # Cleared only for THIS arm's own file: the shared production fuse under $ROOT
  # is not ours to delete.
  local arm_fuse=$dir/serve.fuse
  rm -f "$arm_fuse"

  # Every artifact from this arm lands in $dir; nothing writes outside it.
  local env_line; env_line="$(instrument_env) ${env_delta}"
  # shellcheck disable=SC2086  # deliberate: the helpers emit VAR=val words to split
  ( export SERVE_REPO=$REPO SERVE_ROOT=$ROOT SERVE_PORT=$PORT SERVE_LOG=$arm_log
    export SERVE_FUSE_STATE=$arm_fuse
    setsid nohup env $env_line scripts/serve_hybrid_v100.sh >/dev/null 2>&1 & )
  wait_ready || { log "arm $name: serve never became ready"; return 1; }
  health_gate || { log "arm $name: health gate failed"; stop_serve; return 1; }


  "$PYTHON" scripts/probe_headroom_coldtail.py arm \
    --url "$HEALTH_URL" --headroom 0 --log "$arm_log" --out "$dir/arm.json" \
    --prompt-tokens "$PROMPT_TOKENS" --warm-reps "$WARM_REPS" \
    --expect-window "$WINDOW_TOKENS" >"$dir/arm.log" 2>&1
  local rc=$?
  log "arm $name: probe rc=$rc"

  # The probe's p50 is the WIDE set (dec>0), which keeps captured-graph ticks and
  # the long close-tail ticks; the sweep arms are quoted on
  # dec==1 & sparse==1 & model>0 & sample>0 & path!=graph with the tail listed
  # apart. Re-filter the same log so the two can be tabled together -- or not
  # tabled at all, when this log cannot support the standard set.
  # Window the standard-set re-filter to this arm's WARM spans. Three things must be
  # excluded and a start offset excludes none of them:
  #   - the supervisor's warmup (dense 7000 + sparse 9000 prompts, 8-token decodes)
  #     which runs before any arm measurement;
  #   - EVERY rep's cold FILL, which sits between that rep's warm window and the
  #     previous one in the same log and emits short-context decode ticks that pass
  #     the standard set.
  # The probe's log_byte_offset is a warm START (taken after the fill, before the
  # warm POST), so [off_i, off_{i+1}) would contain warm_i AND fill_{i+1} -- the very
  # ticks this is meant to drop. Each rep therefore carries its own log_byte_end, and
  # every span is an explicit start:end, the last one included (relying on the last
  # rep being clean only because nothing writes after it is step-ordering, not
  # geometry, and a later step added after the filter would silently break it).
  # One --window per rep and ONE call, so the median is taken over the union of the
  # rows rather than averaged across per-rep medians (which would weight a
  # 2-tick rep the same as an 8-tick one).
  local win_args=() spans
  spans=$("$PYTHON" -c '
import json, sys
try:
    arm = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    raise SystemExit          # no arm.json -> no spans -> the branch below says so
bad = []
for r in arm.get("reps", []):
    t = r.get("ticks")
    if not t:
        continue
    start = t.get("log_byte_offset")
    end = t.get("log_byte_end")
    if end is None:
        # No fallback to EOF: only the LAST rep would be clean that way, and only
        # because nothing happens to write after it (step ordering, not geometry).
        # A stale arm.json is refused like a missing one rather than yielding a
        # number that quietly includes the next rep fill.
        bad.append(str(start))
    else:
        print("%d:%d" % (start, end))
if bad:
    print("reps %s lack log_byte_end (arm.json predates the probe fix); the "
          "warm window cannot be bounded, so no steady figure is produced"
          % ", ".join(bad), file=sys.stderr)
    raise SystemExit(3)
' "$dir/arm.json" 2>"$dir/spans.note") || spans=
  if [ -z "$spans" ]; then
    # No spans means no bounded warm windows, so there is NO standard-set figure
    # for this arm. Falling through to an unwindowed (or open-ended) read would
    # write a steady.json whose median contains the supervisor's warmup and a later
    # rep's fill -- a number that looks like the others and is not. Refuse instead
    # of degrading, and surface the extractor's own reason when it gave one.
    log "arm $name: no bounded warm spans in arm.json"
    [ -s "$dir/spans.note" ] && log "  -- $(cat "$dir/spans.note")"
    log "  -- steady.json NOT written: there is no warm span to filter to"
  else
    # Each line is already a start:end span, so the shell only forwards them.
    for s in $spans; do win_args+=(--window "$s"); done
    # ${a[@]+...} so an empty win_args does not trip `set -u` on bash 3.2.
    "$PYTHON" scripts/steady_filter.py --log "$arm_log" --out "$dir/steady.json" \
      ${win_args[@]+"${win_args[@]}"} >"$dir/steady.log" 2>&1
    log "arm $name: steady-filter rc=$? (see $dir/steady.json)"
  fi

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
  #
  # No SERVE_FUSE_STATE either, so this inherits the PRODUCTION fuse under $ROOT --
  # deliberately, unlike an arm. A window that ended on a crash-looping arm can
  # leave that file tripped, and then the official serve exits 2 without starting
  # (the fuse is meant to stay down until an operator arms it again), which would
  # present as "the restore did not come up". Say which it is instead of leaving
  # the reader to guess, and do NOT delete the file: arming the prod fuse again is
  # an operator decision, the same as when the fuse trips on its own.
  local prod_fuse=${SERVE_FUSE_STATE:-$ROOT/.servehybridsse.fuse}
  if [ -f "$prod_fuse" ] && [ "$(wc -l < "$prod_fuse")" -ge "${RESTART_FUSE_MAX:-5}" ]; then
    log "restore: WARNING prod fuse $prod_fuse holds $(wc -l < "$prod_fuse") entries"
    log "  (>= RESTART_FUSE_MAX ${RESTART_FUSE_MAX:-5}); the supervisor will stay down"
    log "  by design. Arm it again with: rm $prod_fuse"
  fi
  ( export SERVE_REPO=$REPO SERVE_ROOT=$ROOT SERVE_PORT=$PORT
    export LIVENESS_POLL_S=60
    setsid nohup scripts/serve_hybrid_v100.sh >/dev/null 2>&1 & )
  wait_ready || { log "restore: serve never became ready (see the fuse warning above)"; return 1; }
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
  # The same three patterns stop_serve signals, so the guard and the stop cannot
  # disagree. Checking only the serve python misses the case that matters: the
  # supervisor RESTARTS a killed python, so a cleanup run between the kill and
  # the reboot would find no `tilerl.cli serve` and delete a spill the incoming
  # boot is about to read.
  local pat
  for pat in "tilerl.cli serve" "serve_hybrid_v100.sh" "serve_liveness.py"; do
    pgrep -f "$pat" >/dev/null 2>&1 && {
      log "REFUSING: a serve is still running ($pat); stop it before deleting the spill"; return 1; }
  done
  # The list arrives on fd 3, NOT on stdin: `done < <(spill_files)` redirects the
  # whole loop's stdin, so the body's `read -r ans` consumes the next NAME off the
  # list instead of the operator's answer and every file is silently "kept"
  # including ones answered y. Keeping the list on its own fd leaves stdin alone,
  # and the here-string keeps quoting (a path with a space survives).
  local f ans files
  files=$(spill_files)
  while IFS= read -r f <&3; do
    [ -e "$f" ] || continue
    ans=
    printf 'delete %s (%s)? [y/N] ' "$f" "$(du -h "$f" | cut -f1)"
    read -r ans
    [ "$ans" = "y" ] && rm -f "$f" && log "deleted $f" || log "kept $f"
  done 3<<< "$files"
}

# ------------------------------------------------------------------------ main
mkdir -p "$OUT"
log "repo=$REPO out=$OUT port=$PORT blocks=$EXPECT_BLOCKS"

if [ "$RESTORE_ONLY" = 1 ]; then restore; exit $?; fi
if [ "$CLEAN_SPILL_ONLY" = 1 ]; then clean_spill; exit $?; fi
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

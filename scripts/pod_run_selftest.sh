#!/usr/bin/env bash
# Selftest for pod_run.sh's runner, on CPU-only paths with card_claim and nvidia-smi mocked.
#
#   scripts/pod_run_selftest.sh     # exits 0, prints PASS
#
# The runner under test is the SHIPPED text: pod_run.sh emits it with
# POD_RUN_EMIT_RUNNER=1. The previous version inlined its own copy, which would have passed
# against any bug in pod_run.sh -- including the one this file now covers.
#
# What it asserts, each having failed on the pod:
#   * a WRAPPER-SCRIPT command still ends up claimed. card_claim refuses a shell pid and
#     that refusal was never retried, so `-- bash wrapper.sh` ran unclaimed and card 6 read
#     ORPHAN with 69583 MiB for 5 minutes (2026-09-07).
#   * an unclaimable job is KILLED, not left running: an unclaimed card is what gets a
#     container restarted under someone else's run.
#   * the claim is released even so.
set -euo pipefail

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/aupai/scripts" "$TMP/work" "$TMP/bin"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Mock card_claim. CLAIM_MODE picks which pod-observed behaviour to replay:
#   shell_then_device -- refuses a shell pid unless --wait-for-device is passed (the real
#                        semantics: it follows descendants until one holds a device)
#   never             -- refuses always, so the kill path is exercised
cat > "$TMP/aupai/scripts/card_claim.py" <<'PY'
import os, sys
log = os.environ["CLAIM_LOG"]
argv = " ".join(sys.argv[1:])
with open(log, "a") as f:
    f.write(argv + "\n")
mode = os.environ.get("CLAIM_MODE", "shell_then_device")
if argv.startswith("acquire"):
    if mode == "never":
        print("pid 123 is a shell, not the job: 'bash /work/wrapper.sh'")
    elif "--wait-for-device" in argv:
        print("claimed 6 for tilerl-selftest")
    else:
        print("pid 123 is a shell, not the job: 'bash /work/wrapper.sh'")
elif argv.startswith("release"):
    print("released tilerl-selftest: 1 claim(s) on 6")
PY

cat > "$TMP/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
# 0 MiB used, so the orphan guard does not trip; index,memory.used for the tail read.
echo "0"
SH
chmod +x "$TMP/bin/nvidia-smi"
export PATH="$TMP/bin:$PATH" CLAIM_LOG="$TMP/claims.txt"

# A wrapper script as the command: the shape whose claim was refused and never retried.
cat > "$TMP/work/wrapper.sh" <<'SH'
python3 -c "import time; time.sleep(1); print('job done')"
SH

fail() { echo "FAIL: $1" >&2; exit 1; }

# macOS has no setsid, and the pod's does not fork (measured: `setsid cmd &` leaves $! as the
# command itself), so the reaping property is the same either way. A shim keeps the shipped
# text runnable here; a Linux run finds the real one first.
command -v setsid >/dev/null || { printf '#!/usr/bin/env bash\nexec "$@"\n' > "$TMP/bin/setsid"; chmod +x "$TMP/bin/setsid"; }

run_arm() {  # run_arm <mode> <outdir> -> writes rc to $2/rc
  local mode=$1 out=$2
  mkdir -p "$out"
  : > "$CLAIM_LOG"
  CLAIM_MODE=$mode AUPAI="$TMP/aupai" REMOTE_DIR="$TMP/work" POD_RUN_EMIT_RUNNER=1 \
    bash "$ROOT/scripts/pod_run.sh" selftest 6 -- bash "$TMP/work/wrapper.sh" > "$out/runner.sh"
  # The runner cds to REMOTE_DIR and logs to /work/<name>.log; redirect both into the tempdir.
  sed -i.bak -e "s#> /work/#> $TMP/work/#g" "$out/runner.sh"
  set +e
  # CLAIM_MODE must reach the RUN, not just the emit: setting it only on the pod_run.sh call
  # left the control using the default mode, and arm 2 passed while proving nothing.
  ( cd "$TMP/work" && CLAIM_MODE=$mode CLAIM_LOG=$CLAIM_LOG bash "$out/runner.sh" > "$out/wrapper.out" 2>&1 )
  echo $? > "$out/rc"
  set -e
  cp "$CLAIM_LOG" "$out/claims.txt"
}

# ---- arm 1: a wrapper-launched job must end up claimed -------------------------------
run_arm shell_then_device "$TMP/a1"
grep -q "claimed 6" "$TMP/a1/wrapper.out" || fail "arm 1: a wrapper-launched job was not claimed: $(cat "$TMP/a1/wrapper.out")"
grep -q -- "--wait-for-device" "$TMP/a1/claims.txt" || fail "arm 1: acquire did not pass --wait-for-device: $(cat "$TMP/a1/claims.txt")"
grep -q "job done" "$TMP/work/selftest.log" || fail "arm 1: the job did not run"
[ "$(cat "$TMP/a1/rc")" = 0 ] || fail "arm 1: rc $(cat "$TMP/a1/rc"), out: $(cat "$TMP/a1/wrapper.out")"
grep -q "^release " "$TMP/a1/claims.txt" || fail "arm 1: release never ran: $(cat "$TMP/a1/claims.txt")"

# ---- arm 3: a multi-arm wrapper re-claims per arm ------------------------------------
# A claim dies with the pid it names, so an N-arm wrapper has N-1 windows where the card
# reads ORPHAN while a later arm runs (seen between two arms of one script, 2026-09-07).
# The rule is that each arm re-claims its own pid; `pod_run_claim` is that call, and this
# arm is what makes it a rule rather than an unused helper.
cat > "$TMP/work/twoarm.sh" <<'SH'
for arm in one two; do
  python3 -c "import time; time.sleep(0.3); print('arm $arm')" &
  pod_run_claim $!
  wait $!
done
SH
: > "$CLAIM_LOG"
mkdir -p "$TMP/a3"
CLAIM_MODE=shell_then_device AUPAI="$TMP/aupai" REMOTE_DIR="$TMP/work" POD_RUN_EMIT_RUNNER=1 \
  bash "$ROOT/scripts/pod_run.sh" selftest 6 -- bash "$TMP/work/twoarm.sh" > "$TMP/a3/runner.sh"
sed -i.bak -e "s#> /work/#> $TMP/work/#g" "$TMP/a3/runner.sh"
set +e
( cd "$TMP/work" && CLAIM_MODE=shell_then_device CLAIM_LOG=$CLAIM_LOG bash "$TMP/a3/runner.sh" \
    > "$TMP/a3/wrapper.out" 2>&1 )
echo $? > "$TMP/a3/rc"
set -e
[ "$(cat "$TMP/a3/rc")" = 0 ] || fail "arm 3: rc $(cat "$TMP/a3/rc"): $(cat "$TMP/a3/wrapper.out")"
# Three acquires: the wrapper's own, then one per arm. Two would mean an arm ran unclaimed.
n_acq=$(grep -c '^acquire' "$CLAIM_LOG")
[ "$n_acq" -ge 3 ] || fail "arm 3: expected >=3 acquires (wrapper + 2 arms), got $n_acq: $(cat "$CLAIM_LOG")"
grep -q "arm two" "$TMP/work/selftest.log" || fail "arm 3: the second arm did not run"

# ---- arm 2 (the control): an unclaimable job is killed, not left running -------------
# Without this arm, arm 1 passes on a runner that ignores the claim result entirely.
run_arm never "$TMP/a2"
[ "$(cat "$TMP/a2/rc")" = 4 ] || fail "arm 2: an unclaimable job must exit 4, got $(cat "$TMP/a2/rc"): $(cat "$TMP/a2/wrapper.out")"
grep -q "card_claim FAILED" "$TMP/a2/wrapper.out" || fail "arm 2: the failure was not reported: $(cat "$TMP/a2/wrapper.out")"
grep -q "^release " "$TMP/a2/claims.txt" || fail "arm 2: release never ran after a kill"

# NOT asserted: that the job is reaped rather than Zs. It cannot fail here -- the zombie needs
# the parent to EXIT while the child runs, leaving it to a PID 1 that never wait()s, and
# macOS's launchd reaps. Verified directly on the pod instead: a child whose bash parent exits
# reads `stat=Zs ppid=1`, the same child under a parent that waits reads reaped. A grep for
# stat=reaped here would pass against a wrapper with the reaping removed.
echo "PASS: a wrapper-launched job is claimed via --wait-for-device; an unclaimable one exits 4 and releases"

#!/usr/bin/env bash
# Selftest for pod_run.sh's runner, on CPU-only paths with card_claim and nvidia-smi mocked.
#
#   scripts/pod_run_selftest.sh     # exits 0, prints PASS
#
# The runner under test is the SHIPPED text: pod_run.sh emits it with
# POD_RUN_EMIT_RUNNER=1. The previous version inlined its own copy, which would have passed
# against any bug in pod_run.sh -- including the one this file now covers.
#
# What it asserts, each having failed on the pod (errors/2026-09-07-a-guard-that-stopped-guarding.md):
#   * a WRAPPER-SCRIPT command ends up claimed -- its python is a descendant.
#   * a DIRECT-PYTHON command is claimed as itself -- it has no descendant to follow.
#   * a multi-arm wrapper re-claims per arm, so no arm runs on an unclaimed card.
#   * an unclaimable job is KILLED, not left running, and the claim is released even so.
#   * a re-claim of a pid pod_run already claimed is a no-op, not a refusal -- acquire says
#     "claim reused, not re-taken" with rc 0 and no "claimed", and a substring test killed the job.
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
# Both refusals are the REAL messages, measured against /work/aupai on 2026-09-07 against a
# python holding no device: --require-device says "holds no GPU device fd", --wait-for-device
# says "no descendant ... opened a GPU device". A mock that granted on the flag alone made
# --wait-for-device look sufficient for every shape, which it is not.
mode = os.environ.get("CLAIM_MODE", "shell_then_device")
# The EXIT CODE is part of the contract the runner reads, so the mock carries it: the real
# card_claim returns 0 on a grant and 1 on a refusal (`return 0 if ok else 1`), and a mock that
# exited 0 for both let a refusal message read as a grant.
if argv.startswith("acquire"):
    if mode == "never":
        print("pid 123 holds no GPU device fd: '/usr/bin/python3 -c ...'")
        sys.exit(1)
    elif mode == "reuse":
        # The pod's real no-op path: pod_run's own block already claimed THIS pid, so an arm's
        # re-claim is a reuse. rc 0, and the word "claimed" never appears -- which is why the
        # runner must test rc. Killed a healthy 27B server on 2026-09-07.
        print("tilerl-selftest already holds 6 for pid 123 "
              "(same pid, same cards -- claim reused, not re-taken)")
    elif mode == "self_device":       # $CMD is python directly: it IS the pid on the card
        if "--require-device" in argv:
            print("claimed 6 for tilerl-selftest")
        else:
            print("no descendant of pid 123 opened a GPU device in 1s -- TIMEOUT")
            sys.exit(1)
    elif "--wait-for-device" in argv:  # a wrapper: the python is a descendant
        print("claimed 6 for tilerl-selftest")
    else:
        print("pid 123 is a shell, not the job: 'bash /work/wrapper.sh'")
        sys.exit(1)
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
python3 -c "import time; time.sleep(${JOB_SECS:-1}); print('job done')"
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
  # JOB_SECS reaches the RUN too: the wrapper reads it when the runner executes it, and the
  # same "set it only on the emit" mistake already cost this file an arm that proved nothing.
  ( cd "$TMP/work" && CLAIM_MODE=$mode CLAIM_LOG=$CLAIM_LOG JOB_SECS=${JOB_SECS:-1} \
      bash "$out/runner.sh" > "$out/wrapper.out" 2>&1 )
  echo $? > "$out/rc"
  set -e
  cp "$CLAIM_LOG" "$out/claims.txt"
}

# ---- arm 0: assembling the runner must not RUN anything on the caller ----------------
# RUNNER_EOF is unquoted, so an unescaped backtick in a runner comment is a command
# substitution evaluated HERE, on the laptop, at assembly time. One in a comment cost a
# `pod_run_claim: command not found` on every launch (2026-09-07); the next one could be a
# word that names a real command. stderr is the only witness -- stdout is still a valid
# runner -- so this arm is the only place it can be caught.
emit_err=$(POD_RUN_EMIT_RUNNER=1 AUPAI="$TMP/aupai" REMOTE_DIR="$TMP/work" \
  bash "$ROOT/scripts/pod_run.sh" selftest 6 -- true 2>&1 >/dev/null)
[ -z "$emit_err" ] || fail "arm 0: assembling the runner wrote to stderr -- an unescaped \` runs on the caller: $emit_err"

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

# ---- arm 4: a DIRECT-python job is claimed as itself, not as a descendant ------------
# The shape every profile run takes: $JOB is the python itself, with no descendant to follow.
run_arm self_device "$TMP/a4"
grep -q "claimed 6" "$TMP/a4/wrapper.out" || fail "arm 4: a direct-python job was not claimed: $(cat "$TMP/a4/wrapper.out")"
grep -q -- "--require-device" "$TMP/a4/claims.txt" || fail "arm 4: acquire never tried --require-device: $(cat "$TMP/a4/claims.txt")"
[ "$(cat "$TMP/a4/rc")" = 0 ] || fail "arm 4: rc $(cat "$TMP/a4/rc"), out: $(cat "$TMP/a4/wrapper.out")"

# ---- arm 5: a re-claim of a pid pod_run ALREADY claimed must not kill the job ---------
# pod_run's own block resolves the wrapper's claim to the descendant python, so a wrapper that
# then calls pod_run_claim on that same python gets acquire's no-op path: rc 0, message "claim
# reused, not re-taken", the word "claimed" absent. A substring test read that as a refusal and
# killed a healthy 27B server after DEVICE_WAIT -- 6 minutes of card 0, 2026-09-07.
JOB_SECS=3 DEVICE_WAIT=2 run_arm reuse "$TMP/a5"
[ "$(cat "$TMP/a5/rc")" = 0 ] || fail "arm 5: a reused claim was treated as a refusal, rc $(cat "$TMP/a5/rc"): $(cat "$TMP/a5/wrapper.out")"
grep -q "claim reused" "$TMP/a5/wrapper.out" || fail "arm 5: the reuse message was not reported: $(cat "$TMP/a5/wrapper.out")"
grep -q "job done" "$TMP/work/selftest.log" || fail "arm 5: the job did not run"

# ---- arm 2 (the control): an unclaimable job is killed, not left running -------------
# Without this arm, arm 1 passes on a runner that ignores the claim result entirely.
# DEVICE_WAIT=2 with a job that outlives it: the condition is "never claimed while ALIVE",
# not "exited before the claim landed" -- a 1 s job under a 300 s poll exits first and takes
# the benign path, which made this arm pass on rc 0.
JOB_SECS=8 DEVICE_WAIT=2 run_arm never "$TMP/a2"
[ "$(cat "$TMP/a2/rc")" = 4 ] || fail "arm 2: an unclaimable job must exit 4, got $(cat "$TMP/a2/rc"): $(cat "$TMP/a2/wrapper.out")"
grep -q "card_claim FAILED, killing" "$TMP/a2/wrapper.out" || fail "arm 2: the job was not killed: $(cat "$TMP/a2/wrapper.out")"
grep -q "card_claim FAILED" "$TMP/a2/wrapper.out" || fail "arm 2: the failure was not reported: $(cat "$TMP/a2/wrapper.out")"
grep -q "^release " "$TMP/a2/claims.txt" || fail "arm 2: release never ran after a kill"

# NOT asserted: that the job is reaped rather than Zs. It cannot fail here -- the zombie needs
# the parent to EXIT while the child runs, leaving it to a PID 1 that never wait()s, and
# macOS's launchd reaps. Verified directly on the pod instead: a child whose bash parent exits
# reads `stat=Zs ppid=1`, the same child under a parent that waits reads reaped. A grep for
# stat=reaped here would pass against a wrapper with the reaping removed.
echo "PASS: a wrapper-launched job claims via --wait-for-device, a direct-python one via --require-device, a multi-arm wrapper re-claims per arm, a reused claim is not a refusal, and an unclaimable job exits 4 and releases"

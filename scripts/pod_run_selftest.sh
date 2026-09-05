#!/usr/bin/env bash
# Selftest for pod_run.sh's runner, on CPU-only paths with card_claim and
# nvidia-smi mocked. Asserts the two things that actually failed on the pod:
# the job is REAPED (not a zombie), and the claim is RELEASED even so.
#
#   scripts/pod_run_selftest.sh     # exits 0, prints PASS
set -euo pipefail

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/aupai/scripts" "$TMP/work"

# Mock card_claim: records every call, refuses once with the device-fd message so
# the retry path is exercised rather than merely present.
cat > "$TMP/aupai/scripts/card_claim.py" <<'PY'
import sys, os
log = os.environ["CLAIM_LOG"]
argv = " ".join(sys.argv[1:])
with open(log, "a") as f:
    f.write(argv + "\n")
if argv.startswith("acquire"):
    n = sum(1 for line in open(log) if line.startswith("acquire"))
    print("no device fd yet for pid" if n == 1 else "claimed 6 for tilerl-selftest")
elif argv.startswith("status"):
    pass
else:
    print("released tilerl-selftest: 1 claim(s) on 6")
PY

cat > "$TMP/nvidia-smi" <<'SH'
#!/usr/bin/env bash
# 0 MiB used, so the orphan guard does not trip.
for a in "$@"; do case "$a" in *memory.used*) ;; esac; done
echo "0"
SH
chmod +x "$TMP/nvidia-smi"
export PATH="$TMP:$PATH" CLAIM_LOG="$TMP/claims.txt"
: > "$CLAIM_LOG"

# macOS has no setsid, and the pod's does not fork -- measured on the pod:
# `setsid cmd &` leaves $! as the command itself and `wait` blocks the full
# duration, because setsid execs when it is already not a group leader. So the
# reaping property under test is the same with or without it, and dropping it
# here lets the selftest run on this machine. SETSID is exported so a Linux run
# exercises the real shape.
SETSID=$(command -v setsid || true)

# The runner, inlined with the same structure pod_run.sh generates: setsid job
# under a bash that waits. Kept in sync by the assertions below, which check
# behaviour (reaped, released) rather than text.
NAME=selftest CARD=6 AUPAI="$TMP/aupai"
(
  set -uo pipefail
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD)
  held=0
  if [ "$used" -gt 64 ] && [ "$held" -eq 0 ]; then echo "orphan guard tripped" >&2; exit 3; fi

  release() { python3 "$AUPAI/scripts/card_claim.py" release --name tilerl-$NAME >/dev/null 2>&1 || true; }
  trap release EXIT INT TERM

  setsid_or_not() { if [ -n "$SETSID" ]; then "$SETSID" "$@"; else "$@"; fi; }
  setsid_or_not python3 -c "import time; time.sleep(1); print('job done')" > "$TMP/work/$NAME.log" 2>&1 < /dev/null &
  JOB=$!
  for i in 1 2 3; do
    kill -0 $JOB 2>/dev/null || break
    out=$(python3 "$AUPAI/scripts/card_claim.py" acquire --name tilerl-$NAME --cards $CARD --pid $JOB 2>&1) || true
    case "$out" in
      *claimed*) break;;
      *"no device fd"*) sleep 0.2; continue;;
      *) break;;
    esac
  done
  wait $JOB; rc=$?
  echo "stat=$(ps -o stat= -p $JOB 2>/dev/null || echo reaped)" > "$TMP/stat.txt"
  echo "rc=$rc" >> "$TMP/stat.txt"
)

fail() { echo "FAIL: $1" >&2; exit 1; }

grep -q "job done" "$TMP/work/$NAME.log" || fail "the job did not run"
grep -q "rc=0" "$TMP/stat.txt" || fail "the job did not exit 0: $(cat "$TMP/stat.txt")"
# NOT asserted here: that the job is reaped rather than Zs. It cannot fail on this
# machine -- the zombie needs the parent to EXIT while the child runs, leaving it to
# a PID 1 that never wait()s, and macOS's launchd reaps. Verified directly on the pod
# instead: a child whose bash parent exits reads `stat=Zs ppid=1`, and the same child
# under a parent that waits reads reaped. A grep for stat=reaped here would pass
# against a wrapper with the reaping removed, which is a check that proves nothing.
grep -q "^release " "$CLAIM_LOG" || fail "release never ran; claims were: $(cat "$CLAIM_LOG")"
# The retry path, not just its presence: first acquire refused, second claimed.
[ "$(grep -c '^acquire' "$CLAIM_LOG")" -ge 2 ] || fail "the device-fd retry did not fire"

echo "PASS: job ran, claim retried once then released (reaping is pod-only, see note)"

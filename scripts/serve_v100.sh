#!/bin/bash
# Serve the 27B NVFP4 on the V100, and come back if it dies. Run detached:
#
#   setsid nohup scripts/serve_v100.sh >/dev/null 2>&1 &
#
# Flags are settled measurements: --depth 1 beats the shipped 3 by 1.204x at B=1
# (#22/#72), --slots 8 because 16 costs 5.19x the KV pool (#65), --max-batch 1
# because B=8 only survives while no prefill is in flight (#42). --max-ctx 32768
# because _fit_blocks returns min(fit, cap) and 32768 keeps CAP the binding one:
# 2048 blocks against a fit measured at 2649 (#65, at the costlier --depth 3).
# 4096 left 8x on the table.
#
# ponytail: bash loop, not a systemd unit -- `loginctl show-user` reports Linger=no
# on this pod, so a --user unit dies with the ssh session.
set -u

ROOT=/data00/home/chenkailun.c
REPO=$ROOT/tilerl-git
LOG=$ROOT/serve70c.log
MAX_RESTARTS=10        # then stay dead: a crash loop is a finding, not a hiccup
LOG_CAP=$((32 * 1024 * 1024))

# One supervisor per pod, else two fight for port 8000 and for the card. flock -n
# fails identically when the lock is held and when flock is missing, so a host
# without it would read as "already running" -- check for the binary separately.
command -v flock >/dev/null || { echo "flock(1) not found; refusing to run unlocked" >&2; exit 2; }
exec 9>"$ROOT/.serve70.lock"
flock -n 9 || { echo "another supervisor is already running" >&2; exit 1; }

cd "$REPO" || exit 1
export PATH=/usr/local/cuda-12.4/bin:$PATH
export TILERL_TARGET=cuda
export TMPDIR=$ROOT/pytmp TMP=$ROOT/pytmp TEMP=$ROOT/pytmp
export PYTHONPATH=$REPO/src:$REPO/packages/tilerl-kernels/src
CKPT=$ROOT/models/Qwen3.8-27B-NVFP4
export TILERL_QWEN38_SOURCE=$CKPT

# Without this, TERM to the supervisor left python holding 23 GiB of the card with
# nothing supervising it -- measured, it needed a second kill by pid.
child=
stopping=
trap 'stopping=1; if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; wait "$child"; fi; exit 143' TERM INT

for ((n = 0; n <= MAX_RESTARTS; n++)); do
  # An unbounded log on this pod is how 123 GiB once filled the disk.
  if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$LOG_CAP" ]; then : > "$LOG"; fi
  echo "=== boot $n at $(date -Is)  sha $(git rev-parse --short HEAD)" \
       "dirty $(git status --porcelain | wc -l) ===" >> "$LOG"
  started=$SECONDS
  # Backgrounded so the trap above can run: bash defers traps while it blocks on a
  # foreground child, which for a healthy server is forever.
  "$ROOT/venv70/bin/python" -u -m tilerl.cli serve --model qwen38-27b \
      --draft "$CKPT/model-00018-of-00018.safetensors" --depth 1 \
      --host 0.0.0.0 --port 8000 --max-batch 1 --max-ctx 32768 >> "$LOG" 2>&1 &
  child=$!
  wait "$child"; rc=$?; child=
  ran=$((SECONDS - started))
  echo "=== exit rc=$rc after ${ran}s ===" >> "$LOG"
  # Only a signal aimed at the SUPERVISOR means stop; the trap sets that flag before it
  # ever reaches here. Reading it off the child's rc instead treats every externally
  # killed server as deliberate -- a plain `kill -TERM <server pid>`, or the OOM killer,
  # exits 143 and the supervisor walked away from exactly the case it exists for
  # (measured: restarting the server to pick up new code left nothing listening).
  if [ -n "$stopping" ] || [ "$rc" = 0 ]; then exit "$rc"; fi
  [ "$ran" -lt 60 ] && sleep 30                   # a load takes ~40s; faster means it never served
done
echo "=== gave up after $MAX_RESTARTS restarts ===" >> "$LOG"
exit 1

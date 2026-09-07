#!/usr/bin/env bash
# Sync this checkout to the H20 pod (tarball over stdin; GitHub is unreachable from the pod)
# and optionally run a command there.
# Usage: scripts/pod_sync.sh ['remote shell command']   # sync, run, wait
#        scripts/pod_sync.sh run <name> 'command'        # sync, detach, poll
#        scripts/pod_sync.sh --session <name> [...]      # sync into that session's tree
# `run` detaches under setsid and polls the log: the connection drops before a 27B bench
# finishes. The script goes over as base64 because a heredoc through `tn exec` arrives empty.
# The wipe below is confined to $REMOTE_DIR, one tree per session (scripts/pod_session.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/pod_session.sh
. "$ROOT/scripts/pod_session.sh"
if [ "${1:-}" = --session ]; then
  POD_SESSION="$2"; shift 2
fi
SESSION="$(pod_session_name "$ROOT")"
REMOTE_DIR="${REMOTE_DIR:-$(pod_session_tree "$ROOT")}"
POD_NAME="${POD_NAME:-sglang-test}"

# the remote checkout is wiped below and the tarball overwrites bench-baseline.json with
# this tree's copy, so a failed pull silently drops any row the pod raised. No `|| true`:
# the sync aborts instead. SKIP_BASELINE_PULL=1 stays the deliberate overwrite.
[ "${SKIP_BASELINE_PULL:-0}" = 1 ] || python3 "$ROOT/scripts/baseline.py" pull >/dev/null
# the pod is not a git repo: stamp HEAD so bench rows carry provenance. Write only when
# git succeeded: `git ... > stamp || true` truncates the file before git runs, so a
# failure left it empty and bench_harness read empty as a blank commit, not "unknown".
if sha=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null); then
  printf '%s\n' "$sha" > "$ROOT/.synced_commit"
fi

# ONE prelude for both entry points. It used to live inside the `run` branch only, so a
# plain `pod_sync.sh 'cmd'` ran the container's tilelang 0.1.8 while `run` ran 0.1.13 from
# the uv venv on /work -- two environments behind one script. `\$PATH` stays literal so the
# pod expands it; a missing /work/tl013 in PATH is ignored, so no `[ -d ]` test.
POD_ENV="export PATH=/work/tl013/bin:\$PATH TILELANG_CACHE_DIR=/work/tilelang_cache"
POD_ENV+=" PYTHONPATH=$REMOTE_DIR/src:$REMOTE_DIR/packages/tilerl-kernels/src"
POD_ENV+=" TILERL_TARGET=cuda REMOTE_DIR=$REMOTE_DIR"

# ~/bin/pod's crictl exec lacks -i (no stdin), so drive tn exec directly.
# tilelang's JIT cache lives on /work: the container's HOME is ephemeral.
# runs/ is exempt: the pod is not a git repo, and since 8388cbf a run writes its
# manifest before the eval arms, so a killed run leaves the one copy that exists.
# Not `-path ./runs -prune -o -delete` -- `-delete` implies `-depth`, which disables
# `-prune`. GNU find refuses that outright (rc=1, nothing deleted); BSD find accepts it
# and deletes runs/ silently, so it reads as protection on a Mac
# (errors/2026-09-06-the-oom-was-micro-zero.md).
wipe="find . -mindepth 1 \\! -path './runs' \\! -path './runs/*' -delete"
inner="cat > /tmp/tilerl-sync.tgz && mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && $wipe && tar xzf /tmp/tilerl-sync.tgz && $POD_ENV${1:+ && $1}"
remote="cid=\$(crictl ps -q --name $POD_NAME --state Running 2>/dev/null | head -1); "
remote+="if [ -z \"\$cid\" ]; then echo 'pod: container not Running' >&2; exit 1; fi; "
remote+="crictl exec -i \$cid bash -lc $(printf '%q' "$inner")"

if [ "${1:-}" = run ]; then
  name="$2"; shift 2
  POD_SESSION="$SESSION" "$0" >/dev/null   # sync this checkout first; the job runs against it
  script=$(printf 'set -x\ncd %s\n%s\necho "pod_sync: tree %s sha %s"\n%s\necho DONE_%s\n' \
                  "$REMOTE_DIR" "$POD_ENV" "$REMOTE_DIR" "$(cat "$ROOT/.synced_commit" 2>/dev/null || echo unknown)" \
                  "$1" "$name" | base64 | tr -d '\n')
  pod_exec() {
    tn exec "cid=\$(crictl ps -q --name $POD_NAME --state Running | head -1); crictl exec \$cid bash -lc $(printf '%q' "$1")"
  }
  pod_exec "echo $script | base64 -d > /work/$name.sh; rm -f /work/$name.log; setsid nohup bash /work/$name.sh > /work/$name.log 2>&1 < /dev/null & sleep 1"
  echo "launched $name; polling /work/$name.log"
  while :; do
    out=$(pod_exec "cat /work/$name.log" 2>/dev/null) || true
    case "$out" in *"DONE_$name"*) echo "$out"; exit 0;; esac
    sleep 30
  done
fi

tar czf - --exclude=.venv --exclude=__pycache__ --exclude=.git \
    --exclude='*.pyc' --exclude='*.egg-info' -C "$ROOT" . \
  | tn exec "$remote"

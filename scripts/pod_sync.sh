#!/usr/bin/env bash
# Sync this checkout to the H20 pod (tarball over stdin; GitHub is unreachable from the pod)
# and optionally run a command there.
# Usage: scripts/pod_sync.sh ['remote shell command']   # sync, run, wait
#        scripts/pod_sync.sh run <name> 'command'        # sync, detach, poll
# `run` detaches under setsid and polls the log: the connection drops before a 27B bench
# finishes. The script goes over as base64 because a heredoc through `tn exec` arrives empty.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# the remote checkout is wiped below; pull any bench row the pod raised first.
[ "${SKIP_BASELINE_PULL:-0}" = 1 ] || python3 "$ROOT/scripts/baseline.py" pull >/dev/null 2>&1 || true
# the pod is not a git repo: stamp HEAD so bench rows carry provenance.
git -C "$ROOT" rev-parse --short HEAD > "$ROOT/.synced_commit" 2>/dev/null || true
REMOTE_DIR="${REMOTE_DIR:-/work/tilerl}"
POD_NAME="${POD_NAME:-sglang-test}"

# ~/bin/pod's crictl exec lacks -i (no stdin), so drive tn exec directly.
# tilelang's JIT cache lives on /work: the container's HOME is ephemeral.
inner="cat > /tmp/tilerl-sync.tgz && mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && find . -mindepth 1 -delete && tar xzf /tmp/tilerl-sync.tgz && export TILELANG_CACHE_DIR=/work/tilelang_cache${1:+ && $1}"
remote="cid=\$(crictl ps -q --name $POD_NAME --state Running 2>/dev/null | head -1); "
remote+="if [ -z \"\$cid\" ]; then echo 'pod: container not Running' >&2; exit 1; fi; "
remote+="crictl exec -i \$cid bash -lc $(printf '%q' "$inner")"

if [ "${1:-}" = run ]; then
  name="$2"; shift 2
  "$0" >/dev/null   # sync this checkout first; the job runs against it
  script=$(printf 'set -x\ncd %s\nexport TILELANG_CACHE_DIR=/work/tilelang_cache PYTHONPATH=%s/src:%s/packages/tilerl-kernels/src TILERL_TARGET=cuda\n%s\necho DONE_%s\n' \
                  "$REMOTE_DIR" "$REMOTE_DIR" "$REMOTE_DIR" "$1" "$name" | base64 | tr -d '\n')
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

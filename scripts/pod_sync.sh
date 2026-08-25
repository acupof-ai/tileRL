#!/usr/bin/env bash
# Sync this checkout to the H20 pod (tarball over stdin — GitHub is unreachable
# from the pod) and optionally run a command there.
# Usage: scripts/pod_sync.sh ['remote shell command']
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/work/tilerl}"
POD_NAME="${POD_NAME:-sglang-test}"

# ~/bin/pod's crictl exec lacks -i (no stdin), so drive tn exec directly.
# Persist tilelang's JIT disk cache on /work (the container's HOME is ephemeral,
# so the default ~/.tilelang/cache dies with the container and re-pays NVCC).
inner="cat > /tmp/tilerl-sync.tgz && mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && tar xzf /tmp/tilerl-sync.tgz && export TILELANG_CACHE_DIR=/work/tilelang_cache${1:+ && $1}"
remote="cid=\$(crictl ps -q --name $POD_NAME --state Running 2>/dev/null | head -1); "
remote+="if [ -z \"\$cid\" ]; then echo 'pod: container not Running' >&2; exit 1; fi; "
remote+="crictl exec -i \$cid bash -lc $(printf '%q' "$inner")"

tar czf - --exclude=.venv --exclude=__pycache__ --exclude=.git \
    --exclude='*.pyc' --exclude='*.egg-info' -C "$ROOT" . \
  | tn exec "$remote"

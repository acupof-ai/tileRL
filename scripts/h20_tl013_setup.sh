#!/usr/bin/env bash
# Build the H20 pod's maintained /work/tl013 interpreter (torch 2.11.0+cu129,
# CPython 3.12, tilelang[fp4]==0.1.13) ONCE in a fresh sglang-test pod.
#
# Why this exists
# ---------------
# The pod image's login python3 is 3.12 with torch 2.11.0+cu129 but tilelang
# 0.1.8, which cannot compile the fp4/fp8 cells. `uv run` is unusable: it builds
# a 3.11 .venv from uv.lock whose torch is 2.13.0+cu130 and the 535 (CUDA 12.9)
# driver refuses it ("driver too old"). The maintained /work/tl013 is a 3.12
# venv with --system-site-packages (so it inherits the image's working cu129
# torch) with tilelang 0.1.13 installed into the venv proper. See
# docs/experience/errors/2026-09-16-h20-pod-uses-tl013-cu129-not-uv-run-cu130.md
# and .../2026-09-09-the-cu129-index-redirect-cannot-run-on-the-pod.md.
#
# /work persists across container restarts, so this runs ONE TIME per pod, from
# inside the pod, before the first pod_run. It does NOT touch a card.
#
#   scripts/pod_sync.sh --session <name> 'bash scripts/h20_tl013_setup.sh'
#   scripts/pod_sync.sh --session <name> 'bash scripts/serve_h20.sh --dry-run'  # verify
#
# Idempotent: if /work/tl013/bin/python3 already passes the version/cuda checks
# it prints OK and exits 0. Override knobs (env):
#   TL013_DIR (default /work/tl013)
#   TL013_PYTHON (default python3; MUST be the 3.12 image interpreter)
#   PIP_INDEX_URL (default https://mirrors.aliyun.com/pypi/simple; full lock incl
#                  huggingface-hub>=1.28 — mirrors.ivolces.com tops out at 1.27)
#   TL013_TILELANG (default "tilelang[fp4]==0.1.13")
#   TL013_FORCE=1 rebuild even if the venv exists
set -euo pipefail

TL013_DIR="${TL013_DIR:-/work/tl013}"
TL013_PYTHON="${TL013_PYTHON:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
TL013_TILELANG="${TL013_TILELANG:-tilelang[fp4]==0.1.13}"
# torch the driver can run. 2.13.x+cu129 cannot (needs nccl>=2.29 the cu129
# index lacks); 2.11.0+cu129 is the known-good pair on the 535 driver.
TL013_TORCH_SPEC="${TL013_TORCH_SPEC:-torch==2.11.0}"
# The cu129 wheel index is reachable from the pod (~15 MB/s); the domestic PyPI
# mirror carries the pure deps. TORCH_INDEX is tried for torch, then the generic
# index for everything else.
TL013_TORCH_INDEX="${TL013_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"

PY="$TL013_DIR/bin/python3"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 2; }; }

# Hard gate the interpreter AFTER install/on reuse: must be 3.12, torch must be
# the cu129 build (NOT +cu130), and CUDA must initialise. Fail closed — a wrong
# env otherwise surfaces as a late "driver too old" after a 27B model load.
verify() {
  "$PY" <<'PYEOF'
import sys, torch
pv = sys.version_info
assert (pv.major, pv.minor) == (3, 12), f"need CPython 3.12, got {pv.major}.{pv.minor}"
assert torch.__version__.startswith("2.11."), f"need torch 2.11.x, got {torch.__version__}"
assert "+cu129" in torch.__version__, f"need +cu129 build, got {torch.__version__} (cu130 cannot run on driver 535)"
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"tl013 OK: py {pv.major}.{pv.minor} torch {torch.__version__} cuda {torch.version.cuda} devices {torch.cuda.device_count()}")
PYEOF
}

if [ -z "${TL013_FORCE:-}" ] && [ -x "$PY" ]; then
  if verify >/dev/null 2>&1; then
    echo "tl013 already present and valid at $TL013_DIR — nothing to do (TL013_FORCE=1 to rebuild)"
    verify
    exit 0
  fi
  echo "warning: $TL013_DIR exists but fails verification; rebuilding" >&2
fi

need "$TL013_PYTHON"
"$TL013_PYTHON" - <<'PYEOF'
import sys
assert sys.version_info[:2] == (3, 12), f"base interpreter must be 3.12, got {sys.version_info.major}.{sys.version_info.minor}; set TL013_PYTHON"
PYEOF

# /work is the persistent pod volume; HOME is ephemeral and rootfs / small.
mkdir -p /work/tmp
# --system-site-packages is deliberate: inherit the image's working cu129 torch
# + nvidia wheel closure rather than re-resolving a torch that may come out cu130.
"$TL013_PYTHON" -m venv --system-site-packages "$TL013_DIR"

# Upgrade pip from the domestic mirror (direct pypi.org stalls on large wheels).
PIP="env PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 $PY -m pip"
$PIP install --index-url "$PIP_INDEX_URL" --upgrade pip

# torch: prefer what --system-site-packages already inherited (the image's
# 2.11.0+cu129). Only pip-install if the inherited build is wrong, and even then
# tolerate an index miss — the final verify() is the hard gate, so an inherited
# good torch wins while a cu130/absent torch is caught explicitly below.
if ! "$PY" - <<'PYEOF'
import torch, sys
sys.exit(0 if (torch.__version__.startswith("2.11.") and "+cu129" in torch.__version__) else 1)
PYEOF
then
  echo "inherited torch is not 2.11.x+cu129, attempting $TL013_TORCH_SPEC from $TL013_TORCH_INDEX" >&2
  $PIP install --index-url "$TL013_TORCH_INDEX" "$TL013_TORCH_SPEC" || {
    echo "could not install $TL013_TORCH_SPEC from $TL013_TORCH_INDEX; set TL013_TORCH_INDEX/TORCH_SPEC" >&2
    verify || exit 3; }
fi

# tilelang 0.1.13 must live in the VENV (shadow the image's 0.1.8), from the
# generic mirror which has the full closure.
$PIP install --index-url "$PIP_INDEX_URL" "$TL013_TILELANG"

# Serve/probe runtime deps not guaranteed by the base image. The project itself
# is never pip-installed on the pod — PYTHONPATH points at the synced tree.
$PIP install --index-url "$PIP_INDEX_URL" \
  numpy safetensors tokenizers "huggingface-hub>=1.28.0" \
  fastapi uvicorn websockets pytest==9.1.0

verify
echo "tl013 ready: $TL013_DIR"
echo "next: scripts/pod_run.sh ... -- bash scripts/serve_h20.sh (pod_run puts $TL013_DIR/bin on PATH)"

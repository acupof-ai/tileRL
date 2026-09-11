#!/usr/bin/env bash
# shellcheck shell=bash
# Source this on the pod to run tilerl without uv: `uv run` builds a .venv whose
# torch is too new for driver 12090 (cuda.is_available()=False, serve hangs silently).
# The container's python3 (3.12) already has torch 2.11.0+cu129 which works.
#   source scripts/pod_env.sh
#   python3 -m tilerl.cli serve ...
_TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$_TREE/src:$_TREE/packages/tilerl-kernels/src${PYTHONPATH:+:$PYTHONPATH}"
export TILERL_TARGET="${TILERL_TARGET:-cuda}"
export TILERL_QWEN38_SOURCE="${TILERL_QWEN38_SOURCE:-/work/tilerl-ckpt/Qwen3.8-27B-NVFP4}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/work/tilelang_cache}"
unset _TREE

# Fail loud: the silent failure mode is a serve that never becomes ready.
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
  echo "pod_env: python3 cannot import torch+cuda; do not use 'uv run' (it builds a .venv with a torch too new for driver 12090)" >&2
  return 1 2>/dev/null || exit 1
}

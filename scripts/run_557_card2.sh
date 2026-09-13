#!/usr/bin/env bash
# #557 card-2 validation, one command. Runs in MY synced tree (the dispatch
# fix): cc's probes are vendored here so the code under test is the PR head.
#   PATH counters + buckets over 64 decode ticks across the 8-tick refresh
#   boundary, sparse-cap vs true sparse-eager token equality, ms/tick for
#   sparse-cap / sparse-eager / dense, matched per-arm prompts (seed 1234
#   re-seeded inside every engine build, so all three arms see one input set).
# Usage: scripts/run_557_card2.sh [probe]   probe: card2 (default) | diag
set -uo pipefail
cd "$(dirname "$0")/.."
export TILERL_TARGET=cuda
export TILELANG_CACHE_DIR=/work/tilelang_cache
export SRC=/work/tilerl-ckpt/Qwen3.8-27B-NVFP4
probe="${1:-card2}"
/usr/bin/python3 "scripts/probe_557_${probe}.py" 2>&1 | tee "/work/v557fix_${probe}.log"

#!/usr/bin/env bash
# V100 (sm70) sparse-KV 256k run with the cold tier spilling to NVMe.
#
# WHY: the 128k/256k sparse run drives the host f16 cold tier to ~16-18 GiB while
# this pod has only ~6 GiB RAM free, so HostKvPages LRU drops cold pages once the
# pinned budget binds (silent re-fetch, lost context). #525 added a one-file mmap
# spill: pages past the pinned host budget move to NVMe and promote back through
# the same seam. This pins the host budget to ~10 GiB and lets the rest spill.
#
# DO NOT LAUNCH FROM HERE — cc's queue owns the V100. Sync and detach through the
# wrapper instead, when the card is free:
#   scripts/v100.sh run v100_256k_ssd 'bash scripts/_v100_256k_ssd.sh'
#
# ---- expected cold-tier bytes (ledger, 27B f16 cold) -------------------------
# 16 full-attn planes x 4 KV heads x 256 head_dim x 16 tokens/page x K+V x 2 B
#   = 1,048,576 B = 1.0 MiB per cold page (per_cold_kv_block_bytes, f16-narrow).
# 256k tokens / 16 = 16,384 pages -> 16.0 GiB if the WHOLE context were cold.
# Host budget pinned here: 10 GiB (10240 pages). Spill to NVMe once the resident
# cold set exceeds that; on a steady decode most pages beyond the k+window hot set
# are cold, so expect up to ~6 GiB (6144 pages) resident in the mmap file at full
# 256k:  kv_cold(host) ~= 10 GiB, kv_cold_ssd ~= 6 GiB (sum ~= 16 GiB).
# The exact split moves with selection; the invariant is host <= 10 GiB and
# host+ssd == demoted-pages * 1 MiB. The mmap file grows a 1 MiB stride per slot.
#
# sm70 specifics (mirrors scripts/serve_v100.sh): f32 device pool narrows its
# demoted K/V to f16 on the D2H (--cold-format f16); no --kv-fp8 (sm70 has no
# fp8 fused writer and no fp8 attention path). --max-batch 1 (B=8 only survives
# with no prefill in flight), sparse bounds scorer (training-free Quest).
set -euo pipefail

ROOT="${V100_REPO:-$HOME/tilerl-v100}"
CKPT="${V100_CKPT:-/data00/home/chenkailun.c/models/Qwen3.8-27B-NVFP4}"
# /data00 is the mounted NVMe-class volume (46 GB free in the v100.sh notes); do
# not use root / (100% full) or /tmp.
SSD="${V100_COLD_SSD:-/data00/sparse_cold_256k.bin}"
LOG="${V100_LOGS:-$HOME/tilerl-logs}/v100_256k_ssd.log"

export PATH=/usr/local/cuda-12.4/bin:$PATH
export TILERL_TARGET=cuda
export TMPDIR=$HOME/tmp
export PYTHONPATH=$ROOT/src:$ROOT/packages/tilerl-kernels/src
export TILERL_QWEN38_SOURCE=$CKPT
export TILERL_MESSAGES_RECORD=$HOME/messages_requests.jsonl

# 10 GiB pinned host (10240 f16 pages); the remainder of the 16 GiB 256k cold set
# spills to $SSD. Sized in bytes so the flag does not depend on page geometry.
HOST_COLD_BYTES=$((10 * 1024 * 1024 * 1024))

# sparse-k 1024 matches the recall-science run's indexed page count; the 8-page
# local window is always added. 262144 cap so the fit binds at the full context.
exec /usr/bin/python3 -u -m tilerl.cli serve \
    --model qwen38-27b \
    --draft "$CKPT/model-00018-of-00018.safetensors" --depth 1 \
    --host 0.0.0.0 --port 8000 \
    --max-batch 1 \
    --max-ctx 262144 \
    --sparse-k 1024 \
    --scorer bounds \
    --cold-format f16 \
    --kv-cold-bytes "$HOST_COLD_BYTES" \
    --cold-ssd-path "$SSD" \
    2>&1 | tee "$LOG"

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
# fp8 fused writer and no fp8 attention path). --max-batch 1 / --slots 1
# (B=8 only survives with no prefill in flight, and the sparse hot pool is
# per-slot), sparse bounds scorer (training-free Quest).
#
# ---- hot-pool sizing: why k=128 and a single slot here ------------------------
# The device hot pool is  slots*(n_groups*k + 8-window + chunk) + 1  f32 pages
# (each K+V f32 page = 2 MiB on the 27B: 16 planes x 4 heads x 16 tok x 256
# x K+V x 4 B). With default chunk=32 pages+1 and 4 source groups:
#   k=128 -> 4*128 + 8 + 33 = 553, i.e. 554 blocks/slot = 1.08 GiB K+V (fits);
#   k=1024 -> 4137 blocks/slot ~= 8 GiB — over the V100's ~4.9 GiB post-weights
#             headroom, and the default 8 slots would be ~32 GiB.
# So this card run uses --slots 1 and --sparse-k 128. k=1024 on V100 is a real
# run only AFTER the engine reserves the UNION of the 4 groups (a shared hot set
# is far smaller than n_groups*k) rather than the sum. That is an engine change,
# not a CLI flag; it belongs in the sparse design doc's measured section.
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

# sparse-k 128 keeps the 4-group union hot pool inside the V100's headroom
# (1107 blocks, 1.08 GiB K+V); see the hot-pool note above. The 8-page local
# window is always added. 262144 cap so the fit binds at the full context.
# No --draft/--depth: sparse_k + speculative draft still raises NotImplementedError
# at build_engine until #530 lands.
exec /usr/bin/python3 -u -m tilerl.cli serve \
    --model qwen38-27b \
    --host 0.0.0.0 --port 8000 \
    --max-batch 1 \
    --slots 1 \
    --max-ctx 262144 \
    --sparse-k 128 \
    --scorer bounds \
    --cold-format f16 \
    --kv-cold-bytes "$HOST_COLD_BYTES" \
    --cold-ssd-path "$SSD" \
    2>&1 | tee "$LOG"

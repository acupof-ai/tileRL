"""Occupancy arithmetic for the sm70 prefill attention tile.

Run this before changing a tile shape. It answers which of Volta's three limits
binds -- shared memory, registers, or warps -- and what a D-blocked variant
costs in K/V traffic.

    python scripts/sm70_tile_occupancy.py
"""

from __future__ import annotations

# Volta (sm70) per-SM limits. The register file is 65536 32-bit registers,
# i.e. 256 KiB -- four times the shared memory, which is why it does not bind.
SMEM = 96 * 1024
REGISTERS = 65536
MAX_REG_PER_THREAD = 255
MAX_WARPS = 64
SMS = 80

D = 256
HEADS = 24
KV_HEADS = 4
FULL_ATTN_LAYERS = 16
CHUNK = 512
COMPUTE_FLOOR_16K = 3.36  # s, causal QK^T+PV at 15.7 TFLOP/s -- the entry's figure
BW = 900e9


def resources(block_m: int, block_n: int, kv_bytes: int, threads: int) -> dict:
    """Shared bytes, registers per thread, and what caps residency."""
    shared = (block_m + 2 * block_n) * D * kv_bytes
    # acc_s + acc_o + the five online-softmax vectors, all f32 fragments,
    # following sm90's make_paged_attention_mma.
    reg_words = block_m * block_n + block_m * D + 5 * block_m
    reg_per_thread = -(-reg_words // threads)
    by_smem = SMEM // shared if shared <= SMEM else 0
    by_reg = REGISTERS // (reg_per_thread * threads)
    by_warp = MAX_WARPS // (threads // 32)
    blocks = 0 if reg_per_thread > MAX_REG_PER_THREAD else min(by_smem, by_reg, by_warp)
    return {
        "shared": shared,
        "reg_per_thread": reg_per_thread,
        "by_smem": by_smem,
        "by_reg": by_reg,
        "blocks_per_sm": blocks,
        "warps_per_sm": blocks * threads // 32,
        "grid": -(-CHUNK // block_m) * HEADS,
    }


def kv_traffic_secs(block_m: int, kv_bytes: int, n: int = 16384, d_block: int = D) -> float:
    """K/V bytes re-read over an n-token prefill, at 900 GB/s.

    d_block < D forces one pass over the same K/V tile per D slice, because the
    online softmax accumulates acc_o over all of D -- that is why D-blocking is
    rejected, not merely disfavoured.
    """
    passes = D // d_block
    total = 0
    prefix = 0
    while prefix < n:
        c = min(CHUNK, n - prefix)
        tiles = -(-c // block_m)
        total += FULL_ATTN_LAYERS * KV_HEADS * (prefix + c) * D * kv_bytes * 2 * tiles * passes
        prefix += c
    return total / BW


if __name__ == "__main__":
    print(f"{'tile':>8}{'kv':>5}{'thr':>5}{'smem':>7}{'reg/t':>7}"
          f"{'by smem':>9}{'by reg':>8}{'blk/SM':>8}{'grid':>6}{'kv_s':>7}  limiter")
    for kv in (4, 2):
        for bm in (64, 32):
            for bn in (32, 16):
                r = resources(bm, bn, kv, 256)
                shared = f"{r['shared'] // 1024}K" if r["shared"] <= SMEM else "OVER"
                lim = "-" if not r["blocks_per_sm"] else (
                    "smem" if r["by_smem"] <= r["by_reg"] else "regs")
                print(f"{bm:>4}x{bn:<3}{'f32' if kv == 4 else 'f16':>5}{256:>5}{shared:>7}"
                      f"{r['reg_per_thread']:>7}{r['by_smem']:>9}{r['by_reg']:>8}"
                      f"{r['blocks_per_sm']:>8}{r['grid']:>6}"
                      f"{kv_traffic_secs(bm, kv):>7.2f}  {lim}")

    chosen = resources(64, 16, 4, 256)
    assert chosen["shared"] == SMEM, "64x16 f32 sits exactly at the 96 KiB limit"
    assert chosen["blocks_per_sm"] == 1
    assert chosen["by_smem"] < chosen["by_reg"], "shared memory binds, not registers"
    assert resources(64, 32, 4, 256)["blocks_per_sm"] == 0, "64x32 f32 is 128 KiB"
    assert resources(64, 16, 2, 256)["blocks_per_sm"] == 2, "rung 2 doubles residency"
    # Every tile from 32 up is far under the compute floor, so traffic stops
    # deciding and occupancy does.
    assert kv_traffic_secs(32, 4) < COMPUTE_FLOOR_16K
    # D-blocking gives back most of the win: 4 passes over every K/V tile.
    assert kv_traffic_secs(64, 4, d_block=64) > 3.9 * kv_traffic_secs(64, 4)
    print("\nchecks pass: 64x16 f32 thr=256, 96 KiB exactly, 1 block/SM, "
          "shared-memory-bound, 192 blocks over 80 SMs")

# sm90 paged attention: FlashAttention port — decode 37.8 -> 0.46 ms, prefill 1056 -> 0.062 ms — H20, 2026-08-24

> Status: Shipped

## Context

The 27B has 16 full-attn layers (every 4th, `full_attention_interval=4`);
slice2 has 0 of them, so full-attn cost was unmeasured. The existing
`make_paged_attention` (kernels.py) is a portable serial-scalar kernel: one
block per (batch, head), serial over query positions and key positions,
scalar FMAs. Measured at the 27B full-attn shapes (H=24, Hkv=4, D=256,
block=16) on H20:

- decode M=1, KV=4096: **37.84 ms** vs 22.5 µs memory roofline (f32 KV) — 0.1%
- prefill M=512, KV=512 causal: **1056 ms** vs 21.8 µs bf16-tensor roofline — 0.0%

Both are >1000x off roofline. The prefill gap is the serial query/key loops
(no tiling, no tensor cores); the decode gap is the serial 256-deep dot
product per key (4096 keys x 256 dependent FMAs per head).

## What Worked

Ported the FlashAttention SOTA schedule
(`examples/flash_attention/example_mha_fwd_bshd.py`) to
`kernels_mma.make_paged_attention_mma`, wired into the sm90 cell (the f32
naive kernel stays in kernels.py as the cpu/metal floor). Adaptations:

- **Paged KV**: the dense `T.copy(K/V)` becomes a block-table gather
  (`KCache[BlockTable[b, p // block_size], hkv, p % block_size, d]`), the
  same indexing as `write_tokens`. Out-of-range gather positions (decode
  padding rows) clamp to the last block and are masked out of the score.
- **GQA**: kv head = `h * Hkv // H`; grid is (query tile, head, batch).
- **bf16 IO, f32 accumulate** (the sm90 cell convention); the backend casts
  at the boundary and pads Q's S dim to a multiple of block_M.
- **block_M as a schedule arg**: 16 for decode (M=1), 64 for prefill. A
  64-row decode tile is compute-bound on 63 garbage rows (intensity 65
  FLOP/byte, above the H20 ridge of 37); a 16-row tile stays memory-bound.
- **16-row tile cast**: with 128 threads the score fragment is replicate-4
  and the bf16 cast-copy conflicts on layout — the cast round-trips through
  shared memory (one writer per element). 64 threads (2 warps) is the
  partition that always works for the PV gemm (4 warps need D a multiple of
  32; the parity test's D=16 breaks it).

Measured (same pod, same GPU, contended — co-tenant at ~99% util, BW ~1.5
TB/s of the nominal 4):

| case | before | after | speedup | roofline | %roof |
|---|---:|---:|---:|---:|---:|
| decode M=1 KV=4096 | 37.84 ms | 0.456 ms | 83x | 11.2 µs (bf16 KV) | 2.5% |
| prefill M=512 KV=512 | 1056 ms | 0.062 ms | 17100x | 21.8 µs (bf16 tensor) | 35.4% |

Prefill at 35% of roofline under contention is consistent with the other
sm90 kernels on this pod (fp8 WGMMA: 16-22% under load, 60-80% idle) — the
kernel is within 2x of roofline on an idle GPU. Decode is still ~30x off the
memory roofline: tilelang 0.1.13 lowers the paged gather to synchronous
loads (no cp_async for elementwise copies), so the kernel is
memory-latency-bound, not bandwidth-bound. Marked ponytail — split-KV
flash-decoding with pipelined per-block T.copy gathers is the upgrade when
decode shows up on the profile.

Full-model impact (16 full-attn layers): decode 16 x 0.456 = 7.3 ms/tick
contended (~2 ms idle est.) vs the GDN slice's 9.3 ms GPU sum — significant
but not dominant; prefill 16 x 0.062 = 1.0 ms per 512-token tick = 0.002
ms/tok, negligible vs the GDN prefill (5657 ms, torch-eager, separate
work).

## Rule

tilelang 1D fragments replicate across all threads (a `T.Parallel(N)` dot
unrolls to N FMAs per thread, icache-bound) — put per-thread vectors in
shared memory to force partition. And `T.Pipelined` peels a prologue that
references loop-carried scalars out of scope (0.1.13 codegen bug) — use
`T.serial` for scalar-carried loops.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | b71c5a3 | H20 | cuda/sm90 | kernel-level (27B full-attn shapes) | 0.062 (M=512) | 0.456 (M=1) | — |

Raw artifacts: `scripts/bench_paged_attn.py` (before/after, `--kernel
naive|mma`).

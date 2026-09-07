# The sm70 prefill tile: 64×16 f32, and no f32 tile gets two blocks per SM — 2026-09-07

> Step 1 of the accepted sm70 prefill cell
> (`wins/2026-09-07-the-sm70-prefill-n2-is-one-kernel-at-30x-its-floor.md`).
> Arithmetic only, no kernel code. **Choice: `block_M=64`, `block_N=16`, f32 K/V,
> 256 threads, full D=256.** Reproduce with
> `python scripts/sm70_tile_occupancy.py`.

## The instruction cannot be met in f32, and that is the finding

The brief was: pick the largest tile that fits 96 KiB with two blocks per SM if
possible, one otherwise. **No f32 tile at D=256 reaches two blocks per SM.** The
constraint is shared memory, and it is tight enough that the largest tile which
fits at all sits exactly at the limit:

```
shared = (block_M + 2·block_N) · D · sizeof(kv_dtype)
       = (64 + 32) · 256 · 4 = 98,304 B = 96 KiB exactly
```

One byte more and it does not fit. Two blocks per SM in f32 would need 48 KiB,
i.e. `block_M + 2·block_N ≤ 48` — at best 32×8, which halves the query tile and
doubles K/V traffic to buy a second resident block. Rung 2 (fp16 tiles) gets two
blocks at 64×16 without giving anything up, which is the argument for it.

## The table

D=256, full-D, 256 threads. Volta: 96 KiB shared/SM, **65,536 registers/SM
(256 KiB)**, 255 registers/thread, 64 warps/SM. Grid at S=512 is
`ceildiv(512, block_M) × 24` heads, against 80 SMs.

| tile | K/V | shared | reg/thread | by shared | by regs | blocks/SM | grid | K/V traffic | binds |
|---|---|---|---|---|---|---|---|---|---|
| 64×32 | f32 | **128 KiB** | 74 | **0** | 3 | **does not fit** | 192 | 0.31 s | — |
| **64×16** | **f32** | **96 KiB** | **70** | **1** | **3** | **1** | **192** | **0.31 s** | **shared** |
| 32×32 | f32 | 96 KiB | 37 | 1 | 6 | 1 | 384 | 0.63 s | shared |
| 32×16 | f32 | 64 KiB | 35 | 1 | 7 | 1 | 384 | 0.63 s | shared |
| 64×32 | f16 | 64 KiB | 74 | 1 | 3 | 1 | 192 | 0.16 s | shared |
| 64×16 | f16 | 48 KiB | 70 | 2 | 3 | **2** | 192 | 0.16 s | shared |
| 32×32 | f16 | 48 KiB | 37 | 2 | 6 | **2** | 384 | 0.31 s | shared |
| 32×16 | f16 | 32 KiB | 35 | 3 | 7 | **3** | 384 | 0.31 s | shared |

**Shared memory binds in every row**; the register file permits 3 to 7 blocks
everywhere and never decides. That is worth stating because it is not obvious
from the fragment shapes: following sm90's `make_paged_attention_mma`
(`kernels_attn.py:54-64`), `acc_s`, `acc_o` and the five online-softmax vectors
are `alloc_fragment`, and `acc_o` at `block_M × D` f32 is 64 KiB per block at
block_M=64 — larger than any shared tile in the kernel, and still only 25% of a
256 KiB register file. Spread over 256 threads it is 70 registers each, well
under the 255 cap.

## D-blocking is rejected, and it is not a close call

D-blocking drops shared memory linearly and would let 64×32 f32 fit. It cannot
be used, because **`acc_o` must stay full-D**: the online softmax accumulates
the output over all 256 dimensions, so a D-slice pass over K/V is a partial dot
product that must revisit the same K/V tile once per slice. Global traffic
multiplies by `D/Dblk` — precisely the traffic the query tile exists to remove.

Bytes read per 16,384-token prefill, 16 full-attention layers:

| scheme | traffic | at 900 GB/s | vs tile 1 |
|---|---|---|---|
| tile 1 (today) | 18.14 TB | 20.16 s | 1x |
| **64×16 f32, full D** | **0.28 TB** | **0.31 s** | **64x** |
| 32×… f32, full D | 0.57 TB | 0.63 s | 32x |
| 64×… f16, full D | 0.14 TB | 0.16 s | 128x |
| 64×… f32, Dblk=64 | 1.13 TB | 1.26 s | 16x |
| 32×… f32, Dblk=64 | 2.27 TB | 2.52 s | 8x |

Dblk=64 gives back three quarters of the win to buy shared memory, and the
second resident block it buys is worth less than the traffic it costs.

## Why 64×16, given every f32 row is 1 block/SM

**Traffic stops deciding above tile 32.** Every full-D candidate lands at
0.16–0.63 s against the **3.36 s compute floor** — from tile 32 upward the
kernel is compute-bound, and 0.31 s versus 0.63 s is invisible behind 3.36 s.
So the choice is made on the compute side.

`block_M=64` halves the query-tile count against `block_M=32`, so each K/V tile
serves twice the rows and the kernel issues half the tile loads. `block_N=16` is
what makes 64 fit under 96 KiB. 192 blocks against 80 SMs fills the card 2.4x
over even at one resident block, which is the number that matters on Volta —
latency is hidden by having more blocks than SMs, not by residency alone.

The honest risk: 8 warps per SM is 12% occupancy. If the measured kernel lands
well above 3.36 s and the profile points at latency rather than bandwidth, the
answer is rung 2, not a smaller tile — a smaller f32 tile costs traffic and buys
no extra resident block at all.

## Rung 2, named now because the layout must not foreclose it

**fp16 K/V and Q tiles in shared memory, f32 accumulators.** At 64×16 it reaches
2 blocks/SM (25% occupancy) with the same query tile, and halves K/V traffic
again. On Volta it also unlocks `mma.sync.m8n8k4` — fp16 in, f32 accumulate —
the tensor-core rung that is otherwise unreachable, since Volta has no bf16 and
`T.gemm` lowers only to that instruction.

So the f32 cell must take the K/V tile dtype as a maker parameter rather than
hard-coding it, and the accumulators must already be f32 (they are, following
sm90). Parity for any fp16 tile stays `allclose(rtol=1e-2)` against
`paged_attention_split` — an fp16 K/V tile is a precision change and gets no
relaxed threshold for being a later rung.

## What this does not answer

The arithmetic is exact from the hardware limits and sm90's fragment shapes.
Three things it does not settle:

1. **Whether TileLang's Volta backend allocates the fragments as modelled.** The
   register figures assume `alloc_fragment` maps to registers with no spill at
   `block_M·D` f32. A spill turns this into a local-memory analysis. The compile
   log is the check, at step 3.
2. **Whether 12% occupancy hides the memory latency.** Modelled as adequate
   because the grid is 2.4x the SM count; unmodelled, and the reason rung 2 is
   named.
3. **Whether `T.Pipelined` lowers usefully on Volta.** Try it, fall back to
   serial, report which — sm90 uses `num_stages=1`, so the ceiling is low.

## Rule

Compute every limit before naming the one that binds. My first pass wrote
Volta's register file as 64 KiB instead of 65,536 registers (256 KiB) and
concluded the tile was register-bound — a headline, an argument and a
recommendation all built on a factor of four. The assertion in
`scripts/sm70_tile_occupancy.py` that the chosen tile reaches 1 block/SM is what
failed and exposed it, which is the case for writing the checks into the script
rather than reading the table.

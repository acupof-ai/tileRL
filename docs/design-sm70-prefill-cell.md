# Approach: an sm70 prefill cell for paged attention

> The design accepted for the sm70 prefill cell, recorded after the fact so the
> reasoning that produced it survives the PRs that implement it.
> Measurement: `wins/2026-09-07-the-sm70-prefill-n2-is-one-kernel-at-30x-its-floor.md`.
> Tile: `wins/2026-09-07-the-sm70-prefill-tile-is-64x16-f32.md`.

## The defect

`backend.py:933` routes **every** sm70 attention call to
`paged_attention_split`, whose grid is `T.Kernel(KVSPLIT, S*H, B)` and whose body
opens `Qf[d] = Q[bb, tt, hh, d]` — **one query row per thread block**. Each block
then streams its slice of the prefix in `block_N=16` tiles.

That is the right shape for what it was built for. Its own docstring says so:
"S enters the GRID, so W verify positions run concurrently", where W is a
speculative verify width, at most `_MAX_VERIFY_W = 8`. At S=8, one row per block
is correct and the split over KV is what fills the card.

A prefill chunk is **S=512**. The same kernel then re-reads K/V once per query
row where a prefill kernel reads it once per query *tile*.

## What it costs

The earlier version of this note gave a 35–211 s traffic bracket and refused to
name a speedup. The profile closed it. On the real 27B, V100, `paged_attention`
per arm:

| n | attention | share of prefill | compute floor | × | bandwidth floor, tile 1 | × |
|---|---|---|---|---|---|---|
| 16,384 | 100.5 s | 50.9% | 3.36 s | **29.9x** | 20.16 s | 4.99x |
| 21,727 | 185.5 s | 59.0% | 5.91 s | 31.4x | 35.18 s | 5.27x |

Two things this settles that the bracket could not:

**The kernel is bandwidth-bound at tile 1, and that is why the gap is 30x.**
Across the two arms the compute ratio climbs (29.9 → 31.4x) while the bandwidth
ratio is nearly flat (4.99 → 5.27x). A kernel whose cost tracks the bandwidth
term and not the compute term is exactly what one query row per block predicts.
The remaining ~5x over the tile-1 bandwidth floor is the schedule's own
inefficiency and is not what this cell is aimed at.

**The expected win is now a number.** Tiling to 64 queries drops the bandwidth
floor to 0.31 s at 16k, so compute at 3.36 s becomes the binding term. Measured
100.5 s against 3.36 s is **97.1 s recoverable, 49.2% of TTFT**. That is the
target; anything from a 6x reduction in the attention term upward is a real win,
and the acceptance gate below is written against the slope rather than this
ceiling.

The KV pool is **f32** on sm70 — `backend.py:353` sets `io = float32` for
`("cpu", "metal", "sm70")` and `engine.py:1503` routes it into `PagedKvPool`,
overriding the bf16 default. Each avoided re-read is therefore 4 bytes, which
doubles what the tile saves.

Scope: **sm70 only.** sm90 already has `paged_attention_mma` with `block_M=64`
for prefill.

## The plan

Four steps, one PR each, in this order.

1. **The shared-memory table** at `D=256`, counting the accumulators, the S and
   P tiles and the online-softmax vectors, for `block_M`/`block_N` in
   {64,32}×{32,16} and for D-blocking. Pick the largest that fits 96 KiB with
   two blocks per SM if possible, one otherwise. If nothing useful fits in f32,
   the escape is **fp16 K/V and Q tiles in shared memory with f32
   accumulators** — named in the table as rung 2 regardless of whether it is
   needed, because on Volta fp16-in-shared also unlocks `mma.sync.m8n8k4`
   (fp16 in, f32 accumulate), so the smem choice must not foreclose the tensor
   core rung. Parity for any fp16 tile stays `rtol=1e-2` against split.
2. **The CPU twin** — `make_paged_attention_prefill` in `kernels.py` with the
   four parity arms plus the tiny end-to-end. No sm70 code in this PR.
3. **The sm70 cell** — routing at `backend.py:933` by `s > _MAX_VERIFY_W`,
   split kept, T=1 and T=8 no-regression.
4. **The acceptance window** on the V100, all three gates.

Then a second kernel step, tiling across the GQA group — planned work, described
at the end of this note.

Steps 1 and 2 have landed and settled two of this note's open questions; where
the text below was wrong, the correction is marked rather than the original
deleted.

## The design

### What Volta lacks, and what that forces

The sm90 cell is `T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)` plus a
second `T.gemm` for PV, in bf16 with f32 accumulators, `block_M=64`,
`block_N=64`, online softmax between them.

On Volta `T.gemm` lowers to fp16-only `mma.sync.m8n8k4` and there is no bf16, so
the MMA family is dead — the registry says this at `registry.py:118`, and it is
borne out: **AST-counted, every sm70-registered maker contains zero `T.gemm`** —
`make_linear_fp4_gemv_sm70`, `make_linear_fp4_gemv_sm70_m`,
`make_gdn_chunk_fused`, `make_paged_attention_split`. (Grep says otherwise; a
line-range grep picks up neighbouring functions in the same file. Count by AST.)

So the cell reuses the sm90 cell's **structure** and none of its instructions:

| sm90 | sm70 cell |
|---|---|
| `T.gemm` QK<sup>T</sup>, bf16 | hand-tiled `T.Parallel` MAC over shared tiles, f32 |
| `T.gemm` PV, bf16 | same |
| online softmax, f32 accum | **unchanged** — this is the part worth copying |
| `block_M=64`, `block_N=64` | `block_M=64`, `block_N=16` (see below) |
| `T.Pipelined(num_stages=1)` | same, if it lowers; serial otherwise |

The online-softmax rescaling (`scores_max_prev`, `scores_scale`, `logsum`) is
arithmetic, not instructions, and it is the subtle part. Copy it verbatim and the
numerics follow the cell that already passes parity.

### Tile shape

`block_M = 64` queries × `block_N = 16` keys, one thread block per
(query tile, head, batch): grid `(ceildiv(S, 64), H, B)`.

**Settled by step 1, and not as this note first guessed.** Shared memory is
`(block_M + 2·block_N) · D · 4`, so 64×32 is 128 KiB against Volta's 96 KiB and
does not fit; 64×16 is **exactly 96 KiB** and does. D-blocking was the obvious
escape and is rejected: `acc_o` accumulates over all of D, so a D-slice pass
revisits each K/V tile once per slice and multiplies traffic by `D/Dblk` — the
traffic the tile exists to remove. No f32 tile at D=256 reaches two blocks per
SM; the register file permits 3–7 everywhere and never binds
(`wins/2026-09-07-the-sm70-prefill-tile-is-64x16-f32.md`).

`KVSPLIT` disappears. Splitting KV across the grid exists to fill 80 SMs when
there is one query row; with 64-query tiles and `ceildiv(512,64)=8` tiles × 24
heads = 192 blocks per layer, the card is full without it.

### Selection

`paged_attention_split` **stays** and stays the decode and verify path. The new
cell is a second registry entry chosen by query width at `backend.py:933`:

```
s > _MAX_VERIFY_W  ->  paged_attention_prefill_sm70   (chunks: S=512)
s <= _MAX_VERIFY_W ->  paged_attention_split          (decode S=1, verify S<=8)
```

The boundary is already the one the split kernel's docstring names, so no new
constant is introduced. A prefill chunk that shares a tick with decode rows
arrives at S<64 and takes the same padded path the sm90 cell uses (`SeqQLens`
masks the padding), or falls to split — the note's open question, decided by
whether the pad wastes more than the re-read costs at that width.

### The CPU twin

Hard gate: every kernel lands in the CPU cell first. `make_paged_attention` in
`kernels.py` is already the f32 tiled reference in spirit but is serial in S
(`for t in T.serial(S)` inside `T.Kernel(B, H)`). The twin for this cell is a new
`make_paged_attention_prefill` in `kernels.py` with the **same tiling structure**
— query tile in the grid, KV tile in the loop — so the CPU path exercises the
same indexing and masking the sm70 cell will, and the parity gate below runs
here on this GPU-less machine.

### The parity gate

`allclose(rtol=1e-2)` against **`paged_attention_split` on the same chunk**, not
against the dense kernel: split is what ships on sm70 today, so it is the
behaviour that must not change. Arms:

1. one chunk at S=512 with prefix 0 — the first chunk, no history
2. one chunk at S=512 with a prefix of 4096 — the case that costs
3. S=64 exactly, and S=65, straddling a tile boundary so the padded tail is
   exercised rather than assumed
4. a ragged prompt whose last chunk is not a multiple of `block_M`

Plus the existing tiny-model end-to-end: same completions from an engine built
with each cell, which catches an indexing error that parity on one tensor misses.

### The measurement that accepts or rejects it

Same probe, same card, same command: `scripts/prof_prefill_ops.py --model
qwen38-27b --tokens 2048,8192,16384,21727 --selfcheck`. Three conditions, all
required:

1. **Slope.** `paged_attention`'s per-chunk cost against prefix, over the
   **16,384 arm** (32 equal chunks, no tail artifact). Today it is 0.0889 →
   6.7584 s, 75.98x. Accept if that ratio falls by at least **6x**; the intercept
   is the chunk's own causal block and is not the target.
2. **Ratio to the floor.** The whole-arm figure moves from **29.9x** toward the
   compute floor. Anything still above ~10x means the tile landed but the
   schedule did not, and the cell is not done.
3. **TTFT.** The 16,384 arm's total falls. Reject if it does not, whatever the
   kernel microbenchmark says.

Plus a serve-side no-regression gate at T=1 (decode) and T=8 (verify), since
those keep the split kernel and must be untouched.

## The planned second step: tile across the GQA group

Not an open question — planned work, and the fix if the residual 5x survives
step 3.

Six query heads share each KV head (H=24, Hkv=4), and a 64-row query tile spans
one head, so it still reads K/V once per head. Tiling over the GQA group as well
lets **one K/V tile serve 6 × block_M rows**, another 6x off the traffic.

It is step 2 of the kernel rather than part of step 1 because it changes the
grid (heads stop being independent blocks) while the query tile does not, and
because the first tile has to be measured before a second is layered on it.

## Open questions

1. ~~**Does 64×32×f32 fit in 96 KiB with D=256?**~~ **Closed by step 1**: it does
   not — 128 KiB — and neither does any f32 tile that reaches two blocks per SM.
   The answer is **64×16, f32, 256 threads, full D**, at exactly 96 KiB and one
   block per SM, and the binding resource is shared memory rather than the
   register file (`wins/2026-09-07-the-sm70-prefill-tile-is-64x16-f32.md`).
2. ~~**Is attention's share on sm70 near the CPU twin's 64%?**~~ **Closed by the
   profile: 50.9% at 16,384 and 59.0% at 21,727, still rising.** The premise
   holds.
3. **Does `T.Pipelined` lower usefully on Volta?** The sm90 cell uses
   `num_stages=1`, so the answer probably does not matter much. Try it, fall back
   to serial, and print which one ran in the acceptance arm line.
4. **What is the remaining ~5x over the tile-1 bandwidth floor?** The cell is
   aimed at the tile, which is the 4x between the two floors, and no cheap probe
   separates the two before the kernel exists. **Gate 2 below is the
   measurement**: if the ratio to the compute floor stays above ~10x after the
   tile lands, the schedule's own inefficiency is what remains, and the GQA
   tiling above is the next lever rather than a smaller tile.

# fp8 KV: approach note

No code yet. This settles the four things that decide the shape of the work, each
read off the tree rather than assumed, and states what the PR must measure.

ckl asked for fp8 KV with dequant at compute time. Nothing of it exists: the
pools are bf16 on sm90 and f32 on sm70 (`kv_cache.py:79-80`, dtype from
`Backend.io`), the GDN state snapshots the same, and the only mention in the tree
is a ponytail line about the **SSD tier** (`kv_cache.py:399-400`), not the HBM
pool.

## 1. Where the seam is: two readers and two writers

`kv_dtype` as an existing maker parameter is **CPU-cell only** — one hit in the
whole kernel tree, `kernels.py:776`, in the CPU `make_paged_attention_prefill`
docstring, added so sm70's fp16-tile variant is one instantiation. No sm90 maker
takes it.

An earlier draft of this note enumerated the three attention makers, concluded "fp8
is a maker parameter, not a second schedule", and **missed the two writers**. Four
sm90 kernels touch the pool:

| maker | KV operands | access |
|---|---|---|
| `make_paged_attention_mma` (`kernels_attn.py:12`) | `KCache`/`VCache` bf16 at `:41-42` | reads, `T.Parallel` elementwise at `:79-80`, `:114-115` |
| `make_paged_attention_decode` (`:124`) | same at `:142-143` | reads, elementwise at `:187` and its V twin |
| `make_paged_attention_combine` (`:239`) | **none** — `PO`/`PM`/`PL` partials only | — |
| `make_write_tokens` (`kernels_mma.py:14`) | `KPool`/`VPool` bf16 at `:27-28` | **writes**, bare scatter at `:40-41` |
| `make_attn_prep` (`:82`) | `KPool`/`VPool` bf16 at `:103-104` | **writes**, `:145-146`, `:153-154` |

The readers are cheap to convert for the reason the earlier draft gave: neither uses
`T.copy` or TMA for K/V, both assign element by element out of the block table, so a
dequant is a multiply on the right-hand side of an assignment that already exists.

The **writers** are what decide the design, and they are converted here as
`make_write_tokens_fp8` and `make_attn_prep_fp8` — separate registry keys selected
per call by the pool's dtype, not per cell. Two things they must do that the readers
do not:

- Take the **raw** plane and the scale plane, not `kv_layer()`'s return. `kv_layer`
  dequantizes into a copy, and a scatter into that copy is discarded with no error.
- `attn_prep` takes its K amax over the **post-RoPE** values. RoPE is a rotation, so
  it preserves the pairwise norm but not the per-element absmax: measured over 2000
  random rows at D=256, RD2=64, the post-RoPE absmax reaches **1.383x** the pre-RoPE
  one, which against a pre-RoPE scale quantizes the worst element to 620 and saturates
  e4m3's 448. K is staged through the rotation in shared memory before the reduction.

`build_engine` still refuses `kv_fp8` on any cell that has a fused writer with no fp8
twin, so an unconverted arch cannot silently drop writes.

Two consequences worth stating before anyone prices this as bigger:

- The elementwise gather is already the documented weak point of the prefill
  kernel (`:17-18`, "the paged gather lowers to synchronous loads, latency-bound
  at M=1"). fp8 halves the bytes that gather moves. It does not fix the
  synchronicity, and a fp8 win measured on top of a latency-bound gather is a
  smaller win than the same change over a TMA path would give.
- `Q` stays bf16. Only the cache is quantized, so there is no query-side rounding
  to gate.

## 2. The scale geometry, stated per axis

The pool is `[num_planes, num_blocks, num_kv_heads, block_size, head_dim]`
(`kv_cache.py:51`, allocated `:79-80`); `BLOCK_TOKENS = 16` (`kv_cache.py:22`) and on the
27B `head_dim = 256` with `num_kv_heads = 4` (`config.py:114-116`) — the attention
head_dim, not the GDN heads' 128 (`config.py:135-138`).

"One scale per block-row" has to name its axes or it is three different designs.
The choice is **one f32 per `(plane, block, head, token)`** — one scale over head_dim's
256 elements, laid out `[num_planes, num_blocks, num_kv_heads, BLOCK_TOKENS]`. It costs
1.56% of the fp8 plane against a per-`(plane, block, head)` grid's 0.098%, so the saving
is **1.969x, not 2.000x** (measured: 33280 B/token against bf16's 65536 at head_dim 256).

**The writer's launch shape decides this, not accuracy.** `make_write_tokens` launches
`T.Kernel(B * S, H)` — one thread block per `(b, t, h)`, with head_dim as the inner
`T.Parallel`. A per-`(block, head)` absmax spans the 16 thread blocks that share a pool
block, so a single-launch fused writer cannot compute it without atomics or a second pass.
A per-token absmax is a reduction over head_dim inside one block. Per-token is the only
grid the fused writer can produce in one launch.

The coarse grid also compounds on partial-block appends, and by a large factor. A block's
scale changes when a later token arrives with a bigger absmax, so every append must
re-quantize the tokens already stored. Three arms on a growing-magnitude fixture (token
`t` at `(1+t)x` token 0's scale), max relative error and token 0's own:

| append rule | max rel | token 0 |
|---|---|---|
| dequantize from the fp8 block, requantize — coarse grid | 0.3376 | 0.3376 |
| bf16 staging buffer per open block, requantize staging — coarse grid | 0.0629 | 0.0584 |
| per-token scale, write only the token written | 0.0588 | 0.0579 |

Token 0 is the worst-hit element in the compounding arm, which is the signature: it is the
one rounded 16 times. Staging fixes it at the cost of 32 KiB per open block per plane.
Per-token needs neither — an append touches its own 256 elements and its own scale, so it
is idempotent by construction. `reference.py`'s self-check asserts a
dequantize-patch-requantize append is **bit-identical** to one-shot quantization, and that
assert reads False on the coarse grid, which is what makes it non-vacuous.

Accuracy is **not** what separates the grids, against this note's first draft. Measured on
tiny (`scripts/probe_kv_fp8_range.py`), both grids give the identical worst element,
5.882e-2 — that is e4m3's 3-mantissa-bit floor, not a grid property. Per-token moves only
the typical element, 2.17e-2 to 2.04e-2, 6%. So the coarse grid was never the accuracy
risk the first draft claimed; it was a write-path hazard.

## 3. Card-only parity, and what the oracle is

No CPU twin is possible: tilelang's C backend has no sub-f32 type — measured
three ways during the fp8 weight port (`float8_e4m3fn` and `bfloat16` both give
`Cannot convert type X to C type`, the same kernel at f32 compiles,
[fp8-frozen-backward-kernel](experience/wins/2026-09-07-fp8-frozen-backward-kernel.md)).
So the parity test skips off sm90 with that reason, exactly as the fp8 weight
backward does.

The oracle is the **bf16 pool through the same kernel**, not a reference
reimplementation: same prompt, same block table, `kv_dtype` the only difference.
Gate on two things, because they fail differently:

- **next-token agreement** over a long prompt — the property serving cares about,
  and insensitive to small logit noise.
- **max relative logit error** — the number that moves first, and the one to
  report per PR so a later change can be compared against it.

One mutant: shift the scale index by one block. That produces finite, plausible
logits and is the failure the coarse geometry invites, so it is the mutant that
proves the gate is not vacuous.

## 4. What the PR must carry

Bandwidth is the claim, so the headline is decode throughput at long context,
where the KV plane dominates the bytes per token:

What fp8 actually saves, from the shipped shape: attention KV is 2 planes x 4 heads
x 256 = 2048 elements a token, so **4096 bytes/token bf16 against 2048 fp8** —
**32.0 -> 16.0 MiB** for one 8k sequence and **128.0 -> 64.0 MiB** at 32k. That is
the bound any tok/s claim has to sit under, and it is small next to the 27B's own
weights: a decode tick reads the weights every token regardless, so the fp8 KV win
is a fraction of the tick, not a halving of it. **The PR states that fraction
before it states a speedup.**

| number | why |
|---|---|
| decode tok/s at 8k and 32k, bf16 pool vs fp8 pool | the headline; decode is bandwidth-bound and the KV plane is what fp8 halves |
| TTFT unchanged | prefill is compute-bound, so fp8 should not move it; a change here means something else changed |
| SSD bytes per entry | the tier's own capacity, and the ponytail at `kv_cache.py:399-400` predicted this |
| max relative logit error + next-token agreement | the accuracy gate, per above |

**Not** `entries_capacity`. An earlier draft of this note asked for it on the grounds that
halved bytes per entry raises it. It cannot move: `_entries_capacity` divides the state
budget by `_snapshot_bytes`, which is the **GDN state snapshot** (`kv_cache.py:1203`), and
never by KV bytes. Read, not assumed — the row was wrong.

KV bytes are computed in **three** places, not one, and they are hand-kept in agreement:
`PagedKvPool.bytes_per_token`, `_fit_blocks` in `engine.py` (which must reimplement it from
`cfg`, because it runs before the pool exists), and `scripts/probe_block_bytes.py`. A change
to the scale geometry has to touch all three.

The GDN state snapshot is **out of scope for the first PR** and may be the larger
prize: on the one measured entry it is **157 MiB against the KV plane's 178** on a
2729-token prompt ([half-the-bytes-were-read-on-the-tick](experience/wins/2026-09-07-half-the-bytes-were-read-on-the-tick.md)),
so it is comparable to the KV plane at that length and does not shrink with a
shorter prompt the way the KV plane does — which is why it binds capacity at short
contexts. It has a different consumer (`gdn_decode_fused`, not attention) and a
different scale geometry question, so folding it in would make one PR carry two
kernels and two gates. It gets its own note once the KV numbers are in.

## 5. Default

Flag first. Flip in the same PR only if the accuracy gate holds **and** the
tok/s gain is real at both context lengths. A flag that ships without a flip is
still worth landing: it is what makes the 32k measurement repeatable by someone
else.

## What this note settles, and what it does not

**e4m3, measured on tiny** (`scripts/probe_kv_fp8_range.py`), against this note's own
expectation that the choice would need the 27B. e4m3 beats e5m2 **1.89x on the worst
element** (5.882e-2 against 1.111e-1) and **2.14x on the typical** one. `zeroed_frac`
is 0.000% in all eight arms at an in-block dynamic range up to 3.1e4, so nothing
underflowed and e5m2's extra range buys nothing it could be paid for with. The 27B gate
can overturn this; e4m3 is the null it has to beat, not an open question.

**Not settled: whether the readers should take fp8 operands at all.** The pool is fp8
and the writers quantize, but `kv_layer()` still dequantizes the whole plane per call
(marked ponytail) and the attention makers still read bf16. That is a correctness
vehicle, not the bandwidth win — the decode tick still moves bf16 bytes out of the
gather. `paged_attention` raises `TypeError` on a raw fp8 plane rather than up-casting
it, because `_dev` would produce values ~448x too small: finite, plausible, and passing
both the token-agreement and logit gates with the scale absent. Converting the readers
is what makes the tok/s claim, and it is the next PR.

**Not settled: what the cascade does to a multi-session number.** A miss publishes ~62
interior entries at a 31k prompt and each evicts another session's shared head
(v100's H20 measurement, entry at 541a37c). `_entries_capacity` divides by the GDN
snapshot, so **fp8 KV is neutral to it** — a flat multi-session hit rate in a bench row
is that property, not an fp8 result.

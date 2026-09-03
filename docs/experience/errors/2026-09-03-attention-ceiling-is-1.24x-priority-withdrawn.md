# Prefill attention: the re-read is real but the prize is 1.243×, V100 sm70, 2026-09-03

> Status: **DEFERRED, and the previous tick's priority is withdrawn.** The mechanism I
> predicted is confirmed — every query row re-reads the whole K/V window, and 512 rows
> in one launch capture only 1.56× of a possible ~512× sharing. But pricing it against
> the profile says attention at **infinite speed** is worth **1.243×** of prefill,
> against 1.57× for a further 2× on the GEMV. "A bigger lever than anything left in the
> GEMV", written last tick, is **wrong**.

## Context

[The prior entry](2026-09-03-prefill-attention-is-what-grows.md) found attention is
19.6% of a 4096 prefill and the only class that grows with context, running at 0.75
TFLOPS with neither roofline binding. I concluded the cost was the kernel's shape and
called it the next lever. This tick measured the shape — and then did the ceiling
arithmetic I should have done first.

## The mechanism, confirmed

`S` is a **grid** dimension (`T.Kernel(KVSPLIT, S*H, B)`), so each query row is its own
block with its own `acc[D]`, walking its slice of the window alone. Prediction committed
in the script before the run: *if the re-read binds, µs/row is flat in S; threshold to
call it a re-read is µs/row at S=512 within 1.25× of S=1.*

| S at ctx=4096 | µs | µs/row | vs S=1 |
|---:|---:|---:|---:|
| 1 | 216.5 | 216.5 | 1.00× |
| 8 | 1282.7 | 160.3 | 0.74× |
| 32 | 4802.9 | 150.1 | 0.69× |
| 128 | 19723.5 | 154.1 | 0.71× |
| 512 | 71001.8 | 138.7 | **0.64×** |

**0.64× — inside the 1.25× threshold, so the re-read is confirmed.** Batching 512 rows
into one launch recovers only **1.56×**, all of it incidental L2 reuse; a Q-tiled block
sharing one K/V load across its rows could in principle capture ~512×. The kernel does
`S × n` work where `n` would do.

Two instruments agree the microbench is measuring the shipped path: 71.0 ms at S=512
over a full 4096 window halves to 35.5 ms at prefill's ~2048 average window, against
**33.7 ms** measured in-model (4310.9 ms / 128 calls) — **1.05×**.

## The factory knobs are already at their optimum

`block_N` and `KVSPLIT` are both compile-time parameters, so the cheap fix was worth
checking before any rewrite. At ctx=4096, S=32 (µs):

| | block_N=16 | 32 | 64 |
|---|---:|---:|---:|
| KVSPLIT=16 | **4002** | 8472 | 24133 |
| KVSPLIT=32 | 4700 | 8691 | 24678 |
| KVSPLIT=64 | 5026 | 9451 | 25185 |

Monotone in both directions and the shipped `(32, 16)` is within 17% of the best cell.
`block_N=64` is 6× worse — the staged `Kf/Vf` fragments are `(block_N, D)` = 64×256
f32 = 64 KB, which does not fit. **No tuning win here; the shape has to change or
nothing does.**

**This table is S=32 only — prefill width.** It is not evidence that KVSPLIT=16 is faster
for a spec tick, which runs S=1 (decode) and S=1+depth (verify); at S=1 the recorded case
*for* 32 is the flat context slope (512→4096 at 157→163 µs,
[wins/2026-09-01](../wins/2026-09-01-sm70-attention-thread-redundancy.md)). Cite the 4002-vs-4700
figure only at prefill width.

Threads reconfirm the earlier fix held: 248 / 158 / 154 / 302 µs at 32/64/128/256t —
flat from 64 to 128, so the redundancy is still gone and only occupancy remains.

## What kills it: the ceiling

Prefill at 4096 is 22382 ms GPU: GEMV 16327, attention 4378, everything else 1677.

| attention speedup | prefill gain | ms/token |
|---|---:|---:|
| 2× | 1.108× | 4.93 |
| 4× | 1.172× | 4.66 |
| 8× | 1.207× | 4.53 |
| **∞** | **1.243×** | 4.39 |

Against the GEMV's remaining headroom on the same mix: **2× → 1.57×, 4× → 2.21×**. A
Q-tiled flash kernel is a new kernel with a new parity gate, an online-softmax rewrite
across a query tile, and SMEM budgeting on a 96 KB/SM card — for at most 1.243×, most
of which needs a 4-8× kernel to reach.

The check that the mix is right: a 1.82× GEMV on it predicts 1.490×, and the shipped
`ncols=2` measured 1.585× end to end. Consistent, so the 19.6% and the ceiling that
follows are sound.

## Rule

**Price the ceiling before measuring the mechanism, not after.** Both numbers came from
the same profile I already had. I spent a tick confirming a mechanism whose maximum
value I could have computed in one line first — and had published "the next lever" on
the strength of it being 42× off peak. **Distance from peak is not prize size**; a term
at 19.6% of the tick cannot pay more than 1.243× however badly it is written.

Second: **a knob sweep is the cheapest possible refutation of "this needs a rewrite".**
Two factory parameters, nine cells, one run — and it established that the shipped
configuration is near-optimal, so no rewrite-free win existed. That is worth knowing
before *and* after the ceiling arithmetic.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | µs/row, S=1 → S=512 @4096 | 216.5 → **138.7 (0.64×)** |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | sharing captured by S=512 batching | 1.56× of a possible ~512× |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | microbench vs in-model per call | 35.5 vs 33.7 ms (1.05×) |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | best (KVSPLIT, block_N) of 9 | (16, 16) 4002 µs vs shipped 4700 |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | thread sweep 32/64/128/256 | 248/158/154/302 µs |
| 2026-09-03 | (this) | V100 32GB | cuda sm70 | qwen38-27b | **prefill gain at infinite attention** | **1.243× — defer** |

Reproduce: `scripts/prof_attn_ctx.py` (parity, context, S sweep, threads, knob grid).

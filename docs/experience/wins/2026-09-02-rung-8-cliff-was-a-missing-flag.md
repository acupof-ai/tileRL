# The rung-8 cliff was a missing flag — V100 (sm70), 2026-09-02

> Status: Shipped. Prefill 3.3-3.6×; time-to-first-token at 4096 ctx 127 s → 38 s.

## Context

Everything about speculation on this card was shaped by one claim: the sm70 fp4
GEMV serves M ∈ {1,2,4,8} at 22-45 µs/row, and M>8 "falls off a cliff" to 127
µs/row, so width 9 costs ~14.5× width 8. That number rejected tree verification,
rejected wider speculative blocks, capped `spec_depth` at 3, and framed the
staircase as a property of Volta.

It was a property of our dispatch. `backend.py` passed X pre-packed as f16
(`xh=True`) only on the M≤8 branch. The M>8 branch called the same factory
without it — and the packing is the documented reason the ladder is fast at all
("packing collapses 127 µs/row flat to 24-45 µs/row"). The extern is
`tl_fp4_gemv_tiles_f16_m_xh<G, M>`, templated on M with **no upper bound**.

## What Worked

Pass the flag above 8 too.

| shape | M | shipped µs/row | packed µs/row | gain | absdiff |
|---|---:|---:|---:|---:|---:|
| 17408×5120 | 32 | 122.4 | **29.3** | 4.18× | 0.00e+00 |
| 5120×17408 | 16 | 124.4 | 35.7 | 3.49× | 0.00e+00 |
| 5120×17408 | 32 | 127.7 | 30.8 | 4.15× | 0.00e+00 |

**Bit-exact**, which is why the microbench alone was enough to ship on: both paths
round X to nearest f16, so a nonzero difference would have meant the packed extern
reads the wrong bytes at that M — not a precision trade to adjudicate.

At M=32 packed X is 29-31 µs/row against rung 8's 22-45. The staircase is still
real (a rung still rounds up) but the cliff between 8 and 32 is gone.

lm_head could not be measured: `pack_fp4` wants 40 GB of **host** RAM at
N=248320. A harness limit, not a kernel one.

The two dispatch branches collapse into one chunked loop that picks the rung per
chunk (1/2/4/8/32), `LADDER_WIDTHS` gains 32, and the engine's batch warning is
rewritten — it used to warn about a per-row penalty that no longer exists, so it
now reports launches per layer instead.

## End to end: all of it lands on prefill, none on decode

| ctx | prefill before | after | ms/prompt token | gain |
|---:|---:|---:|---:|---:|
| 512 | 15488 ms | **4277** | 30.25 → 8.35 | 3.62× |
| 1024 | 31089 | 8727 | 30.36 → 8.52 | 3.56× |
| 2048 | 62646 | 18016 | 30.59 → 8.80 | 3.48× |
| 4096 | 127446 | **38383** | 31.11 → **9.37** | 3.32× |

Decode is unchanged at every context (dense 37.6 tok/s at 4096, spec 49.7 at
1024). That is not a disappointment, it is the shape of the result: B=1 decode is
M=1 and a speculative verify is M=B×W=4, so **neither ever reaches M>8**. Prefill
is M=512 on every layer of every chunk, and it took the entire win.

Before this, prefill cost **31 ms per prompt token against decode's 26.6** — a
512-row forward was more expensive per token than a single-row one. The arithmetic
said it ran as if M=1: one prefill forward was 15.9 s, 512 rows decoded one at a
time would be 13.6 s, and 16 chunked launches re-streaming 14.40 GB each should be
0.3 s. It is now 9.37 ms/token, finally cheaper than decode.

(The "prefill 15.1 s at 4096" in `wins/2026-08-31-sm70-gdn-chunk-fused.md` is a
**warm-cache** number — a prefix hit. `scripts/bench_prefill.py` uses distinct
tokens per rep so nothing is served from the prefix store. Different quantities,
not a contradiction.)

## Three paths were paying this

- **Prefill.** M=512 chunks at 32 rows, every layer, every chunk. Took the win.
- **Batched verify.** The rung is chosen on ROWS, and a verify tick submits B×W of
  them. `max_batch=4` at depth 3 is M=16 — already over the flag's old ceiling and
  silently on the unpacked kernel. The existing ladder guard could not see it: it
  compared `1 + spec_depth` with no batch term. Untested at B>1.
- **Anything wider.** Every "we cannot go past 8" argument, including the tree and
  top-k verdicts, was priced against 127 µs/row.

## Rule

When a cost model has a cliff in it, check whether the cliff is in the hardware or
in the branch that dispatches to it. A 14.5× discontinuity between adjacent sizes
is not what a memory system does — it is what a code path does. The tell was
available for free: the fast branch and the slow branch called *the same factory*
with different flags, and the flag's own comment said it was worth 5×.

Second, and this nearly cost the change: **"the kernel got 4× faster" and "the
system got faster" are separated by how many times per token that path runs.**
Decode runs it zero times. `bench_ctx_decode.py` excludes prefill from its window
by design, so it reported a flat result, and the honest reading of that alone is
"the microbench win does not exist end to end" — followed by a revert. The number
that mattered needed a harness that did not exist yet
(`scripts/bench_prefill.py`). Before believing a no-op, check that the benchmark
can see the path you changed.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-02 | (this) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | **9.37** | 26.6 | 37.6 dense @ 4096 |
| 2026-09-02 | 036d6b7 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 31.11 | 26.6 | 37.6 dense @ 4096 |

Raw artifacts: `scripts/ab_gemv_xh_m32.py` (per-shape µs/row),
`scripts/bench_prefill.py` (the table above).

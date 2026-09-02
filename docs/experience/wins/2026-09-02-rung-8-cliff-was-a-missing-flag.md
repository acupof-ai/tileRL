# The rung-8 cliff was a missing flag — V100 (sm70), 2026-09-02

> Status: Shipped (microbench conclusive, end-to-end pending in the same session)

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

## Three paths were paying this

- **Prefill.** M=512 chunks at 32 rows, every layer, every chunk.
- **Batched verify.** The rung is chosen on ROWS, and a verify tick submits B×W of
  them. `max_batch=4` at depth 3 is M=16 — already over the flag's old ceiling and
  silently on the unpacked kernel. The existing ladder guard could not see it: it
  compared `1 + spec_depth` with no batch term.
- **Anything wider.** Every "we cannot go past 8" argument, including the tree and
  top-k verdicts, was priced against 127 µs/row.

## Rule

When a cost model has a cliff in it, check whether the cliff is in the hardware or
in the branch that dispatches to it. A 14.5× discontinuity between adjacent sizes
is not what a memory system does — it is what a code path does. The tell was
available for free: the fast branch and the slow branch called *the same factory*
with different flags, and the flag's own comment said it was worth 5×.

Second: a claim that shapes many decisions deserves re-derivation when it becomes
load-bearing again. This one was measured once, correctly, on the unpacked kernel,
and then quoted for two days as though it described the hardware.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-02 | (this) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | pending | pending | pending |

Raw artifacts: `scripts/ab_gemv_xh_m32.py`.

# sm70 fp16-twiddle GEMV: decode 19.9 -> 26.4 tok/s — V100/sm70, 2026-08-30

> Status: Shipped

## Context

The sm70 fp4 GEMV decoded natural nibbles with a branch-free bit-synthesis
(~10 ops/elem, issue-bound at 28-38% roofline). The fp16-twiddle permutes the
packed bytes offline so each nibble lands in an fp16 exponent/mantissa field;
one `mul.f16x2` by 2^14 rebiases the e2m1 exponent, cutting decode to ~1.9
ops/elem. sm70 has no `cvt.rn.f16x2.f32` (sm80+) and no `mul.bf16x2`, so the
sm90 bf16-twiddle twin is dead here — this is the fp16 twin, its C extern
(`T.call_extern` + `T.import_source`) doing straight-from-global loads.

## What Worked

- **fp16-twiddle decode** (`tl_fp4_decode8_f16`): prmt + shift/mask + 4x
  `mul.f16x2` by 0x74007400 (2^14 per lane). Bit-exact vs the e2m1 LUT
  (numpy sim, 50k words; `tests/test_fp4_twiddle.py`).
- **sm70 GEMV** (`make_linear_fp4_gemv_sm70`): GROUP=4 tiles of 16 elem per
  thread, fp16 accumulate inside the 16-elem scale block, one f32
  scale-accumulate per tile. Split-K (reduce_thread=32, block_K=512).
- **Eager twiddle in `materialize`** (not lazy in `_served_fp4`): the twiddle
  allocates a same-size scratch; by the first forward the KV cache +
  activations have left no room on a 32GB card. Tagged `_tl_layout` so
  `_served_fp4`, `save_hf`, and re-materialize skip it.
- **M>1 prefill fix (the regression)**: `materialize` twiddles ALL `.wq`
  weights, but sm70 M>1 has no twiddle-aware kernel — it fell to the generic
  `linear_fp4` which decodes NATURAL nibbles, silently feeding it twiddled
  bytes. The prefill corrupted the KV/hidden, so decode output gibberish.
  The M=1 GEMV was always correct (real-weight parity passed); only M>1 was
  broken. Fix: sm70 M>1 loops the twiddle-aware M=1 GEMV per row (correct,
  M launches, prefill-only). An untwiddle-copy OOMs here (the forward's GPU
  is full — the same scratch constraint that forced eager twiddle).

## M-row GEMV (B=8 decode)

The M=1 GEMV streams W once per token. At B=8, eight M=1 launches re-stream
the same W eight times — 8× the weight bytes, 8 launches/layer, and the
per-row loop OOMs at B=8 (eight concurrent Res + pad scratch tensors live
across the loop). The M-row twin (`make_linear_fp4_gemv_sm70_m`) loads and
decodes WQ ONCE per tile and reuses it across M=8 rows in the C extern
(`tl_fp4_gemv_tiles_f16_m`): one `ld.global.nc.v2.u32` + one decode per
tile, then an inner M-loop of `ld.global.nc.v4.f32` + 8× `fma.rn.f16x2`
per row. Weight bytes — the bottleneck — no longer scale with M.

- **M-row kernel** (`tl_fp4_gemv_tiles_f16_m<G,M>` + `tl_warp_reduce_m_f16<M>`):
  W loaded+decoded once per tile, reused across M rows. `T.const("N, K")`
  bakes the shapes; M is a compile-time factory arg (M=8). The M-loop is
  NOT `#pragma unroll`-ed: unrolling 8 rows inside the G-unroll (4×)
  spills registers.
- **Backend dispatch** (`linear_fp4`): sm70 + twiddled layout, M≤8 → the
  M-row kernel (one launch, W shared). M>8 (prefill) keeps the per-row
  M=1 GEMV loop.
- **Parity**: M=1 AND M=8 vs f32 reference on 12 real checkpoint
  projections. ALL PASS (worst 6.17e-4, gate 1e-2).
  `scripts/parity_real_weights.py`.

## The 1.3 -> 31.8 tok/s trap (pod code drift)

The first B=8 e2e measured 5.67 s/tick (1.3 t/s, 9% GPU) — the kernel was
fast (micro-bench: 4.5× per-token win over M=1) but the tick was CPU-bound.
Root cause: the pod's `engine.py` was an OLD copy (admit ONE waiting request
per tick via `if`, not `while`). Each tick admitted one more request,
producing a mixed prefill+decode tick — mixed ticks skip the decode-graph
path, so every tick ran eager with JIT re-compilation. Syncing the pod's
code to the worktree (the `while` loop was already committed) fixed it:
all 8 requests admit in tick 1, the B=8 graph captures once on tick 2, and
ticks 3+ replay at 263.8 ms.

## Rule

A twiddle that rewrites served bytes must be matched by EVERY kernel that
reads them, or the untwiddled path silently decodes garbage. The M=1 parity
gate was too narrow — the M>1 prefill path is a separate decode contract.
Test every M-dispatch bucket on real checkpoint weights, not just pack_fp4
synthetic ones (the micro parity passed on both, but only the e2e caught the
M>1 generic-kernel mismatch).

A second rule: the bench box is part of the bench. A stale engine.py on the
pod cost a debugging session that looked like a kernel problem. `rsync` the
tree before measuring; the kernel is innocent until the code is proven in
sync.

## Results

| date | commit | machine | target | model | B | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-08-29 | (prev) | V100 | cuda/sm70 | 27B NVFP4 | 1 | — | 48.0 | 19.9 |
| 2026-08-30 | 7a39f37 | V100 | cuda/sm70 | 27B NVFP4 | 1 | — | 37.9 | 26.4 |
| 2026-08-31 | (this) | V100 | cuda/sm70 | 27B NVFP4 | 8 | — | 263.8 | 31.8 |

B=1: steady-state decode (per-step timing, first 3 ticks skipped — the first
is 51s JIT+capture, the "548s-in-the-window trap"). Graph capture SUCCEEDS
with the C-extern kernel (no fallback warning). Correctness: "The capital of
France is" -> " Paris. The capital of Germany is Berlin. The capital of
Italy is Rome...".

B=8: 8 concurrent requests, continuous batching packs them into one forward
per tick. Graph captures once (B=8 bucket), replays at 263.8 ms/tick = 8
tokens/tick -> 31.8 t/s aggregate. Per-token cost 33.0 ms (vs B=1's 37.9 ms)
— the M-row kernel's weight sharing makes B=8 cheaper per-token than B=1.
All 8 requests correct ("Paris", "Jupiter", "Shakespeare", "Au", "Everest",
"yen", "Portuguese"). `scripts/bench_decode_b8.py`.

Micro-benchmark (large-K GEMV): 35% -> 55% MBU. Effective HBM: 475 GB/s =
53% of the V100's 900 GB/s roofline (was 42%).

M=8 micro-benchmark (`scripts/bench_sm70_gemv_m8.py`): N=4864,K=4864 —
453 µs per call (56.7 µs/token) vs M=1's 252.7 µs. The 4.5× per-token win
is the batching effect: W bytes shared, only X bytes and FMA scale with M.

**Physics ceiling**: W4A16 = 0.75 B/param (0.5 WQ + 0.25 scale) -> ~18 GB/token
-> 900 GB/s gives ~50 tok/s max for B=1. 60 tok/s is not reachable on V100 with
W4A16; 26.4 is 53% of the 50 t/s ceiling. For B=8 the weight bytes are shared
across 8 rows, so the per-token ceiling is ~18GB/8 = 2.25 GB -> 900 GB/s gives
~400 tok/s max; 31.8 is 8% of that — the gap is kernel launch overhead,
attention, and sampling, not weight bandwidth.

Raw artifacts: `scripts/bench_decode_steady.py`, `scripts/parity_real_weights.py`,
`scripts/bench_sm70_gemv_f16.py`, `scripts/bench_sm70_gemv_m8.py`,
`scripts/bench_decode_b8.py` (V100, GPU 0, JIT-cached).

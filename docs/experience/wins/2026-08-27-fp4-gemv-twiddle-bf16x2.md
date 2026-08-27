# fp4 GEMV: twiddle decode + bf16x2 FMA — cuda(H20), 2026-08-27

> Status: Shipped (sm90 cell; CPU/metal keep the natural nibble layout)

## Context

The shuffle-LUT fp4 GEMV issued 6.8 instr/elem and was issue-bound at 82%
(errors entry same date). agent-infer's gap (84.5 vs our 54 tok/s) is this
kernel: theirs runs the weight stream near roofline.

## What Worked

- **Twiddled weight layout** (`reference.twiddle_fp4`): tilelang's
  `decode_fp4_to_bf16_twiddling` bit layout, plus a slot permutation so the
  decoded bf16x2 pairs line up with natural bf16x2 X words (no X shuffling).
  Applied once in place on sm90 (`Backend._served_fp4`, flagged tensor);
  `save_hf` untwiddles by the flag; CPU round-trip test.
- **`tl_fp4_gemv_tile16`** (C, `T.call_extern`): decode 2 words (36 ops),
  8× `fma.rn.bf16x2`, unpack + one f32 scale-FMA per 16-elem scale block.
  bf16 accumulation is confined to one block: relerr 3.8e-3 vs 2.0e-3
  (gate 1e-2). GROUP=2 chunks loaded before decoding.
- Same-process A/B, 100 iters, real 27B shapes (`ab_fp4_twiddle`, deleted):

| shape | shipped µs | twiddle µs | Δ |
|---|---:|---:|---:|
| gate_up 34816×5120 | 83.5 | 56.8 | −32% |
| qkv 14336×5120 | 66.4 | 39.2 | −41% |
| down 5120×17408 | 48.2 | 39.9 | −17% |
| o_proj 5120×6144 | 39.7 | 39.5 | eager floor, see errors |

- The MMA paths (prefill, B=8 w4a8) decode the same bytes with the same
  intrinsic inside `_dequant_fp4_macro` (`tl_fp4_decode8_p` -> natural-order
  bf16, then the float scale). A first cut used a TIR bit-gather (~12 int
  ops/elem) and cost B=8 -10% / prefill -6% — the dequant is NOT fully hidden
  behind the WGMMA at decode M=8; the harness gate caught it. With the
  intrinsic: B=8 +29%, prefill +13% — the old nibble+bitcast dequant was a
  real cost there too.

## Rule

Dequant cost is instructions per element; the LUT shuffle's "1 op/elem" was
3 with extraction and sat on the same issue budget as the FMAs. Decode to
packed bf16x2 and use packed FMA. The fp8 GEMV (40% of the B=1 tick) wants
the same treatment (e4m3→bf16 needs a cvt, not a shift).

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | 94a43eb | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | 0.64 | 16.2 (B=1, d512) | **61.7** B=1 d512 (+14%); B=8 agg **184.7** (+29%); prefill 1792 (+14%); 57.7 @2k, 45.8 @8k, 25.2 @32k |

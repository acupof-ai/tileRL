# fp4 packed scales: f32 -> e4m3 — native dtype, 4x less scale traffic, decode regression — REVERTED — H20, 2026-08-25

> Status: Reverted (e4m3 is a 5-11% decode regression; f32 scales restored in ea8ba7f)

## Context

The internal fp4 packed format stored per-32-block scales as f32 (4 bytes/block).
The NVFP4 checkpoint's native scale dtype is e4m3 (1 byte/block) — the f32
conversion at load was pure overhead, and the scale stream was ~15% of the
decode weight traffic (0.125 of 0.625 bytes/elem). This change stores the
scales as e4m3fn bytes (uint8 view), cutting scale traffic 4x. The open
question was whether the e4m3 decode (extra instructions per scale) hurts the
issue-bound decode GEMV more than the traffic savings help.

## What Worked

- **The format change is correct and parity-green.** `pack_fp4` rounds
  block_max/6 to e4m3; `unpack_fp4`/`dequant_fp4` decode via a uint8->e4m3
  view (torch-side mirror of the in-kernel decode). The three sm90 fp4 kernels
  (`make_linear_fp4_mma`, `make_linear_fp4_gemv`,
  `make_linear_fp4_fp8_mma`) and the CPU `make_linear_fp4` read uint8 Scale
  and decode in-register. Zero byte = 0.0, so the zero-padded K-tail still
  kills padded WQ bytes. CUDA parity 31/31 green (kernel vs reference with the
  same e4m3 bytes); local 72 tests green.

- **The decode is integer bit synthesis, not exp2.** The first cut used
  `exp2 + T.if_then_else` (~17 ops per micro-tile), costing 10-26% on the
  decode GEMV. Replacing with the e2m1fn-style bit-trick (normal values are
  pure integer ops reinterpreted as float, subnormals keep the select, sign
  bit not decoded — scales are positive magnitudes) recovered the regression
  to 7-11%. Verified bit-exact vs torch's e4m3 decode for all 127 positive
  byte values (0x7F=NaN decodes as 480.0 — scales never hit it).

- **Per-linear GEMV (clean, idle pod, BW 3308, direct kernel):**

  | shape (N,K) | f32 ms | e4m3 ms | e4m3/f32 |
  |---|---:|---:|---:|
  | 5120,17408 | 0.0602 | 0.0643 | 1.07x slower |
  | 17408,5120 | 0.0612 | 0.0678 | 1.11x slower |
  | 248320,5120 (lm_head) | 0.5202 | 0.5777 | 1.11x slower |

  The GEMV is issue-bound (33-55% of roofline), so the e4m3 decode
  instructions cost more than the 15% traffic savings. The roofline % drops
  (33%->22%, 55%->35%) partly because the roof itself is tighter (0.53125 vs
  0.625 bytes/elem).

- **End-to-end is a +6% decode regression, NOT neutral.** Slice4 graph-captured
  decode (30-tick avg, fully-idle pod, all 8 GPUs at 0%, BW 3312): f32 1.828
  ms/tick (547 tok/s) vs e4m3 1.937-1.945 ms/tick (514-516 tok/s) — +6.1%.
  The lm_head alone (+11%, 0.5195 -> 0.5765 ms) is 28% of the slice4 wall and
  contributes +3.1%; the per-layer GEMVs (+7-11% direct) contribute the rest.
  The earlier "1.821 -> 1.841, neutral" measurement was an anomaly — it is
  inconsistent with the per-linear regression above (lm_head +11% alone forces
  >= +3% on the wall). Slice4 prefill-512: 0.0557 vs 0.0565 ms/tok (neutral,
  prefill is compute-bound). Reverted in ea8ba7f; the e4m3 work stays in git
  history if a memory-bound config appears.

- **Precision (e4m3 vs the old f32 scales) BLOWS rtol=1e-2 on a tiny model
  forward** — reported, not silently loosened. Per-linear scale rel err
  3.4-5.9% (tiny random weights -> subnormal e4m3 scales, ~5% typical);
  end-to-end logits allclose(rtol=1e-2) = False (max abs diff 4.7 on
  magnitude-30 logits). A normal-range control (weights x100) also blows
  (11.9/31.6) — this is e4m3's inherent ~5% worst-case per-block precision,
  not a decode bug (kernel vs reference with the same bytes is exact). The
  real 27B model is trained with e4m3 scales natively (NVFP4 checkpoint), so
  this is the checkpoint's native precision, not a serving regression — the
  f32 scales were more precise than the checkpoint's own format.

## Rule

e4m3 block scales are the checkpoint's native dtype and cut scale traffic 4x,
but they trade issue for traffic: the in-register decode adds instructions
that cost 7-11% on an issue-bound decode GEMV, and the regression surfaces
end-to-end (+6% on slice4 decode — lm_head alone is 28% of the wall and forces
>= +3%). Do NOT ship e4m3 scales on the decode GEMV while it is
issue-throughput-bound; the f32 scale (one direct load per micro-tile) is
faster. The bit-trick decode (integer reinterpret, no exp2) is mandatory if
e4m3 is ever revisited — the exp2 version is 2x worse. e4m3's ~5% per-block
scale precision is inherent; don't compare against f32 scales as if they were
the precision target (the checkpoint's native format is e4m3).

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | 306a8bf (f32 scales) | H20 idle | cuda/sm90 | 27B slice4 (3 GDN + 1 FA) | 0.0557 | 1.828 (wall) | 547 decode |
| 2026-08-25 | 6548450 (e4m3 scales) | H20 idle | cuda/sm90 | 27B slice4 (3 GDN + 1 FA) | 0.0565 | 1.937 (wall) | 516 decode |

Decode ms/tick is the graph-captured per-tick wall (30-tick avg,
`scripts/profile_slice.py --decode-graph`), both arms in the same fully-idle
window (all 8 GPUs at 0%, BW 3312 GB/s). Per-linear GEMV numbers above are
direct-kernel (no backend overhead) from `scripts/bench_fp4_gemv.py`. The e4m3
arm is +6.1% on decode (regression); reverted in ea8ba7f.

Raw artifacts: pod `/work/gemv_roof_f32_clean.log` (f32 per-linear),
`/work/gemv_roof_e4m3_bittrick.log` (e4m3 per-linear),
`/work/slice4_f32_clean.log` (f32 slice4), `/work/slice4_e4m3_v2.log`
(e4m3 slice4), `/work/parity_e4m3.log` (31/31 CUDA parity).

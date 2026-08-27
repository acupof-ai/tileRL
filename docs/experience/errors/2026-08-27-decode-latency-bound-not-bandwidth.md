# Decode is latency/kernel-count-bound, not bandwidth-bound — cuda(H20), 2026-08-27

> Status: verdict (measured). Kills the scale-dtype lever; redirects perf to
> kernel-count reduction.

## Context

27B logits are correct (zero-centered RMSNorm fix). Decode B=1 graph tick =
19.1 ms/64 layers = 52 tok/s; Arle = 84.5 (11.8 ms). The recorded "#1 lever"
(fp4 scale f32→bf16) assumed bandwidth-bound. Measured it before shipping.

## What the measurement said

`scripts/bench_fp4_gemv.py` + `bench_gemv_gap.py`, real per-layer shapes, H20
(BW measured 3254 GB/s):

**Every GEMV floors at ~0.065–0.073 ms regardless of size.** A 63 MB linear
(5120×17408) and a 0.16 MB one (48×5120 = GDN `ab`) take the SAME time:
- fp4 big linears: **~30% roofline** (0.068 ms vs 0.021 ms roof)
- fp8 linears: **13–26% roofline**
- tiny linears (48×5120): **0.1% roofline** — pure launch latency
- direct kernel (no backend): still only **46–49% roofline** on the big ones

Bytes are not the bottleneck. Cutting scale bytes ~8% cannot move a tick that
is ~30% of the byte roofline. **Scale f32→bf16 is dead** (measured, not argued).

**Backend Python overhead is 1.2–1.4× (big) / 1.7× (small) in EAGER** — but the
shipped decode path captures `model.forward` into a CUDAGraph
(`engine.py:191`), so `_rows`/`_pad2d`/`_plan`/`_dev` do NOT run during replay.
The gap is eager-only, already erased in production (graph replay is 4× the
eager gpu-sum). So backend-trim is not the lever either.

## The actual bottleneck: kernel count

~99 captured kernels/tick at 8 layers → **~792/tick at 64 layers, ~24 µs
average each**. The tick is the sum of per-kernel launch/occupancy floors, not
weight traffic. Biggest fusible populations per 64 layers:
- **rmsnorm ~168** (input + post_attn + q_norm + k_norm) — each a separate launch
- **add ~128** (residual adds) — fuseable into the preceding op's epilogue
- **state_gather + state_scatter ~96** (GDN state plumbing)
- rope ~32, silu_mul ~64

## Rule

For this decode tick, latency/occupancy dominate, not HBM bandwidth. The lever
is **fewer, larger kernels** (fusion: norm→GEMV prologue, residual-add
epilogues, GDN state gather/scatter into the chunk kernel), NOT byte reduction
(scale dtype) and NOT backend Python trim (the graph erases it). Any future
"reduce bytes" idea must first clear the ~30% roofline bar — below it, bytes are
free.

## Levers, re-ranked by this data

1. **Fuse rmsnorm into the consuming GEMV** (or at minimum fuse q_norm+k_norm
   into one launch) — ~168 kernels → fewer, and removes an HBM round-trip.
2. **Residual add as a GEMV/attention epilogue** — kills ~128 launches.
3. **GDN state gather/scatter into the chunk kernel** — ~96 launches.
4. Attention/GDN kernel occupancy (n_partition/reduce_thread) — the ~30% floor
   on the big GEMVs is the second-order ceiling once count is cut.

Dead: scale f32→bf16 (bytes), backend Python trim (graph-erased), SR
register-B (a bandwidth/register lever — same roofline bar).

## Results

| date | machine | tick B=1 (64L) | tok/s | GEMV %roof | verdict |
|---|---|---:|---:|---:|---|
| 2026-08-27 | H20 gpu7 | 19.1 ms | 52 | 30% (fp4), 13–26% (fp8) | kernel-count-bound |

Raw: `/work/bwbench.log`, `/work/prof.log`. Instruments: `scripts/bench_fp4_gemv.py`,
`scripts/bench_gemv_gap.py`, `scripts/profile_decode_tick.py`.

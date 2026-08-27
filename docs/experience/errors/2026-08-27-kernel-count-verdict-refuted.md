# "Kernel-count-bound" was wrong: decode is GEMV-bound — cuda(H20), 2026-08-27

> Status: verdict (measured, in-graph). Supersedes the lever ranking in
> `2026-08-27-decode-latency-bound-not-bandwidth.md`.

## Context

Yesterday's verdict: ~785 kernels/tick × ~24 µs → "fewer launches" was lever #1.
Acted on it: `attn_prep` fuses q_norm + k_norm + RoPE + KV-write into ONE launch
per full-attn layer (~11 launches → 1, ~176 fewer per tick).

## Root Cause

**Cutting 176 launches bought 0.3 ms (+2%).** The 24 µs "average" was
measured in EAGER with cuda events — it prices launch+sync, which the decode
CUDA graph already erases. Profiled INSIDE the graph replay
(`scripts/profile_graph_kernels.py`, 8 layers, B=1, 321 kernels/tick):

| kernel | n/tick | µs each | share |
|---|---:|---:|---:|
| linear_fp8_gemv | 17 | 68 | 40% |
| linear_fp4_gemv | 22 | 45 | 34% |
| ~280 small kernels (rmsnorm/add/copies/attn/gdn) | | 1–5 | 26% |

At B=8 the WGMMA decode kernels are 80% (182 µs + 143 µs each) — 3–4× the M=1
GEMV for the same weight stream.

## Fix

Kept `attn_prep` (+2%, correctness green). Redirected the work to the GEMVs:
grouped-prefetch fp8 GEMV (−8% on 16480×5120, see wins entry). Tick 19.3 →
18.4 ms with both.

## Rule

Price a kernel where it runs: **inside the graph, per-kernel GPU time from
torch.profiler over replays** — not eager cuda-event means, which include the
launch/sync the graph removes. Small elementwise kernels cost 1–5 µs in a
graph; fusing them is a ~2% lever, not a 40% one. The 74% (B=1) / 80% (B=8)
that is GEMV/WGMMA at 30–63% of roofline is the lever. Also: **one timing job
per host** — another tenant's nvcc on the same host inflated B=8 ticks 60%
(57 → 92 ms) with no GPU contention at all; the harness now stamps loadavg.

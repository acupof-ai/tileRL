# 27B KV fp8 on H20/sm90: device-verified correct and 1.969x capacity, still no decode speed-up — 2026-09-16

> Status: **Device arm F21 passed.** fp8 KV is correct and usable on sm90; stays a
> flag, **default off**, because at the batches we serve (B≤8) there is no throughput
> benefit — measured decode is 0.83–0.95x. Enable for capacity (long-context density),
> not speed.

This entry closes the single measurement the 2026-09-07 fp8 entry left open — the
**per-tick decode ratio at equal concurrency** — on the real 27B, so the throughput
verdict is no longer an end-to-end-at-unequal-rows estimate. Probe
`scripts/probe_kv_fp8_27b.py`, tree 8428babd, `/work/tl013` (torch 2.11.0+cu129),
H20 card 2 (`scripts/pod_run.sh --lend-ref …`), 2048-token prompt, 24 new tokens,
bf16 pool as the reference. Run exited 0; card released to 0 MiB and independently
re-verified. Raw JSON: pod `/work/kvfp8-27b.log`.

## 1. Accuracy — passed

Next-token agreement of the fp8 pool against the bf16 pool through the same engine:

| metric | value |
|---|---:|
| next-token agreement | **24 / 24 = 1.0** |
| first divergence | null |
| K / V max round-trip error over row amax, **e4m3 per-token** | **3.57% / 3.57%** |
| K / V, e4m3 block-head grid (not chosen) | 5.88% / 5.88% |
| K / V, e5m2 per-token (not chosen) | 7.14% / 11.11% |
| zeroed fraction, e4m3 per-token / block-head | ~9.0e-6 / ~1.2e-5 |

Confirms the tiny verdict on the shipping model: e4m3 with one f32 per-token scale
is the right grid (1.65x better than block-head, 3.1x better than e5m2); real 27B K
amax p50 5.66 / max 22.9 sits inside e4m3's range, nothing underflows. The accept
condition is the 24/24 greedy match; 24 tokens is a short window and this claims
nothing about long-run generation quality.

## 2. Capacity — 1.969x

| | bf16 | fp8 | ratio |
|---|---:|---:|---:|
| KV bytes per token | 65536 | 33280 | 1.969x |
| blocks fitted @32k, B=32 boundary arm | 45338 | 89281 | **1.969x** |
| peak requests resident (of 32) | 22 | **32** | 1.455x |

Block ratio matches byte ratio to four decimals — the fit is exactly proportional.
bf16 was block-bound (45338 / 2048 blocks-per-32k-row = 22 = its peak resident); fp8
held the whole batch of 32 and would fit 43, so its 1.455x resident ratio understates
the gain and **1.969x blocks is the capacity number**. This is what the flag buys.

## 3. Throughput — slower at the batches we serve

Per-tick decode, fp8 vs bf16, at equal concurrency (the previously-open number):

| cell | KV share of a decode tick (bf16 → fp8) | decode ceiling | **measured decode ratio** | decode ms/tick bf16 → fp8 |
|---|---:|---:|---:|---:|
| B=1 @8k | 2.1% → 1.1% | 1.011x | **0.949x** | 20.5 → 21.7 |
| B=1 @32k | 8.1% → 4.3% | 1.041x | **0.847x** | 21.8 → 25.7 |
| B=8 @8k | 14.9% → 8.2% | 1.079x | **0.915x** | 30.5 → 33.3 |
| B=8 @32k | 41.3% → 26.3% | 1.255x | **0.831x** | 21.5 → 25.8 |

Whole-run (prefill+decode) ratios are prefill-bound and also under 1: 0.94x
(B1@8k) down to 0.80x (B8@32k).

The byte-model ceiling says a gain is *possible* only once KV dominates a decode
tick — 1.011x at B1/8k rising to 1.255x at B8/32k. The measured ratio sits below
1.0 in every cell: the fp8 readers dequantize at every gather (a cast + multiply
per KV element), about **+20% per decode tick** (21.5→25.8 ms at B8/32k), and that
ALU cost exceeds the bytes saved while weights dominate. Weights are **24.4 GB
resident**, so at B≤8 KV is at most 41% of the tick — there is not enough KV share
for the halving to win. Crossover needs a larger batch/context where KV is the
majority of a decode tick (the ceiling reaches 1.570x at B=32/32k); that shape is
not served today and the boundary arm was capacity-only, not a rate run.

## Verdict

**sm90 device verification passed: the fp8 KV path is numerically correct and
usable, and it doubles resident KV capacity. It is not enabled by default because
at current batch sizes decode is 5–17% slower and prefill is slower too — the flag
is a capacity lever for dense long-context serving, to switch on when a workload
is KV-capacity-bound at large batch, not a general speed-up.** Audit finding **F21
closes** with this arm; F22 (Quest bounds CPU twins) is unaffected and remains open.

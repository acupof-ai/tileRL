# An improving server read as a degrading one — V100 (sm70), 2026-09-02

> Status: no defect in the engine. The served rate **improves** 1088 → 26.2
> ms/token over six requests and lands within 1.27× of the bench. I reported it
> as a 6.5× regression that degrades per request. Both halves were wrong, from
> two instrument errors and one invalid comparison.

## Context

Warm `serve` looked like 7.83 tok/s where `bench_ctx_decode` reads 48.4 at the
same context and depth. Four sequential HTTP requests then read 7.83 → 5.61 →
2.10 tok/s, which I called an accumulation and wrote up as a 6.5× gap with a
prime suspect (the 144 MiB GDN snapshot the prefix store publishes).

## Root Cause

**The four requests generated different token counts: 120, 120, 120, 42.** I
divided tokens by wall clock for each and compared the rates as if the work were
equal. It is not — a 42-token request pays a fixed per-request cost over a third
as many tokens, so its rate is lower for a reason that has nothing to do with
time. The "degradation" was the token count falling, nothing else.

Two instrument errors compounded it:

1. **The first profiler timed from `submit`**, so the prompt's prefill sat inside
   ms/token: 256 tokens × 7.89 ms = 2020 ms, i.e. 31.6 of a reported 64.9. Half
   the "gap" was prefill charged to decode.
2. **It re-submitted one prompt**, which is a prefix-cache hit on every repeat —
   the opposite of what serve does. So it could not have reproduced the effect it
   was built to explain, in either direction.

Corrected — decode only, distinct prompts, equal token counts — the picture
reverses completely:

| request | ms/token | tok/s | free GiB |
|---:|---:|---:|---:|
| 0 | 1088.2 | 0.9 | 2.90 |
| 1 | 388.3 | 2.6 | 2.90 |
| 2 | 347.2 | 2.9 | 2.81 |
| 3 | 120.0 | 8.3 | 2.81 |
| 4 | 39.9 | 25.0 | 2.81 |
| 5 | 26.2 | 38.1 | 2.81 |

`last/first = 0.02×`. Free HBM is flat, so nothing leaks — which also refutes the
snapshot suspect on its own terms, as did its counter: `prefix_published` read 1
per request where the hypothesis needed 4, and one 150 MiB clone is 0.003
ms/token against a 1088 ms token, three orders of magnitude out.

Every point on that curve is a CUDA graph capture. A speculative tick captures
one graph per (batch bucket × chain width); the width varies per tick because the
draft's confidence truncates each chain, so depth 3 at `max_batch` 2 needs up to
8 graphs, and until one exists that tick *is* the capture rather than a replay.

## Fix

`serve` warms up, which it never did — `bench_ctx_decode.timed` has warmed since
three benchmarks were ruined by exactly this. The first version of the warmup was
also wrong (one request, `depth + 2` tokens): a single request only exercises the
widths it happens to use. It now submits `max_batch` rows of distinct prompts and
generates 48 tokens each, then prints `warm in Ns`. `--no-warmup` opts out.

`--max-batch` also drops 8 → 2, which suits a single-user endpoint and halves the
grid to capture.

## Rule

**A rate is only comparable across equal work.** Four requests of 120/120/120/42
tokens produce four rates that cannot be lined up, and the artifact points the
same direction a real regression would. Compare totals, or fix the denominator
before comparing anything.

Second: **when a measurement disagrees with a benchmark, suspect the measurement
of measuring something else.** Both errors here — prefill inside the window, a
prompt that hits the cache — made serve look slower in ways that had nothing to
do with serve. The 6.5× was ~1.27× once the instrument matched.

Third, and the one I should have applied unprompted: **check a hypothesis against
the counter that already exists before building anything.** `prefix_published`
was in `/health` output I had already read; it said 1 where my story needed 4.
The arithmetic (0.003 ms/token vs a 1088 ms token) was available at the same
moment and refuted the suspect three orders of magnitude over.

Fourth: **a curve needs three points.** With four unequal readings I fitted a
narrative to a direction. Six equal ones show a monotone warmup that is not
ambiguous at all.

## Results

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---:|
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | request 5, decode only | **26.2 ms/tok** |
| 2026-09-02 | (this change) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | request 0, cold | 1088.2 ms/tok |
| 2026-09-02 | (reference) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | bench_ctx_decode 512 d3 | 20.7 ms/tok |
| 2026-09-02 | (wrong) | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | "warm serve, degrading" | 127.7 ms/tok |

Also ruled out on the way, and worth keeping: the daemon thread and HTTP cost
nothing. The same engine driven by a direct `step()` loop and by
`engine.run()` + `take()` agree to 0.2% (64.9 vs 65.0 ms/token).

Raw artifact: `scripts/prof_serve_gap.py`.

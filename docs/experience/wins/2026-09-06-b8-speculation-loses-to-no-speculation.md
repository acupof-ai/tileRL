# Speculation loses to no speculation at B=8, and the loss decomposes into three near-equal thirds — H20 sm90, 2026-09-06

> Status: Shipped (measurement; no default changed)

## Context

At B=1 speculation wins on this model. The open question was B=8, where the
verify tick's `M = B × W` rows cross launch buckets: does the **draft** cost or
the **kernel step** into a wider launch explain the loss? A draft-depth sweep
alone cannot answer it — every arm it can run has a draft attached.

Config for every number below: ctx 2048, wikitext ×8 passages, 128 max_new_tokens,
B=8, card 6, `# probe f49d50ff89b1 / 0f0613f60cd4, engine tree d1f2bb8, arch sm90`.

## What Worked

**The measurement that mattered was the arm the sweep cannot produce.** W=1 needs
an engine built with **no draft** — `--depths 0` is refused at
`ab_draft_depth.py:343`, and `engine.py:405` rejects width ≤ 1 whenever a draft
is attached. `scripts/ab_w1_baseline.py` builds that engine and imports
`_build_model` / `measure` / `bucket` / `wikitext_ids` / `BLOCK_TOKENS` from the
sweep, so the config is copied rather than re-chosen.

| arm | ms/tick | tok/fwd | tok/s | vs no-spec |
|---|---:|---:|---:|---:|
| **W=1, no draft** (mma8) | 23.34 | 6.90 | **295.6** | — |
| W=2, depth 1 | 39.47 | 11.53 | 292.0 | 0.988x |
| W=4, depth 3 | 58.32 | 16.22 | 278.2 | 0.941x |
| W=8, depth 7 | 66.82 | 16.85 | 252.1 | 0.853x |

**Speculation is a net loss at B=8 at every depth measured.** The sweep's own
line reads "best depth 1 at 292.0 tok/s", which is best *among speculative arms*;
the arm that beats all seven is the one it has no flag for.

### The draft cost is not a constant per forward

Timed with CUDA events, ms per draft forward **falls** as depth rises: 5.19 (d1),
4.74, 4.37, 3.97, 3.95, 3.53, 3.74 (d7). Fitting `d_ms = a/n + b` on the
endpoints:

**a = 1.68 ms fixed per tick, b = 3.50 ms per forward**, max residual 0.39 ms
(11% of b) across all seven points.

Quoting a single per-forward figure taken at d1 overstates the marginal cost by
48%. Two independent estimators agree on b to **1.8%**: the within-bucket
difference `(71.45 − 60.43)/3 = 3.67` ms on wgmma64-only ticks (29 at d4, 30 at
d7) against the event-timed 3.74 at d7.

### The decomposition

On the wgmma64-only d7 tick, using the measured W=1 base:

| term | ms | share |
|---|---:|---:|
| draft, `a + 7b` | 26.15 | 36.6% |
| W=1-equivalent base (measured) | 23.34 | 32.7% |
| kernel step mma8 → wgmma64 | 21.96 | 30.7% |
| **tick(W=8)** | **71.45** | |

Three near-equal thirds, not one dominant term. Cross-check: base + kernel step
= 45.30 ms against the probe's independently computed verify figure of **45.73**
— **0.9%**. That agreement is what makes the split quotable rather than three
numbers that happen to sum to the total.

The consequence: eliminating the draft entirely still leaves 21.96 ms of pure
width penalty, which is why the FR-Spec ceiling stops where it does.

## Bounds, not measurements

FR-Spec (one block-parallel draft forward instead of seven): 71.45 → 49.40
ms/tick, **1.446x**, break-even at keeping **69.1%** of the autoregressive head's
tok/forward (11.65 tok/fwd against the 16.85 measured here). This is an **upper
bound** — it assumes a parallel head drafts as well as an autoregressive one,
and a parallel position cannot see what was sampled before it.

## Not established

- **The depth ranking.** Best-to-worst across the seven speculative arms is
  1.158x, and d1-vs-d3 is 1.050x — but this run had `--prompts 8 --batch 8`, i.e.
  **one** disjoint group, against a recorded 15.8% acceptance variance between
  wikitext passages. The ranking is inside passage noise and is **not called**.
  It would need `--prompts 24` for three groups.
- **Nominal width is realized in 58% of ticks.** At d7, 52 ticks split wgmma64
  30 / wgmma32 10 / mma88 7 / wgmma16 5 — `verify_lens` truncation,
  block-boundary clamping and the tick's pad-up all move it. Every figure above
  keyed to a bucket uses **realized** buckets; a claim keyed to nominal W is
  wrong for 42% of ticks.
- **B=8 only, one context, one corpus.** Nothing here says where the crossover
  against B=1's win lies.

## Instrument notes

Four instrument defects, all found by a failure rather than by reading:

- **A monitor that reported "no rows" for an entire successful run.** Its regex
  never matched the row format; the sweep had been printing results throughout.
  A silent monitor said nothing about the job, only about the pattern.
- **A tok/s column 8x too high** (2364.1 for a 27B on one card). The formula
  multiplied by batch, but `measure`'s tok/fwd is already batch-aggregate — found
  by reproducing all seven sweep rows to ±0.1 with no batch factor. Caught
  because 8x is not physical; at 1.3x it would have shipped.
- **`BLOCK_TOKENS` retyped as 256.** It is **16** (`kv_cache.py:22`), so the KV
  pool was sized 13x short and the run died at `exceeds KV pool capacity`. Now
  imported.
- **`_build_model` reads `TILERL_QWEN38_SOURCE`, never `--source`.** Without it
  the run reaches for the Hub and dies on a network error. `ab_draft_depth.py`
  carries the same latent dependency and only worked because the pod shell had
  the variable set; both now derive it from `--source`.

## Rule

A sweep's best arm is not the best arm. `ab_draft_depth.py` can only build
engines that speculate, so its winner is the best *speculative* config — and at
B=8 the no-speculation arm beats all seven of them. Before quoting a sweep's
optimum, ask which configurations it structurally cannot express, and measure one
of those.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | d1f2bb8 | H20 card 6 | cuda sm90 | 27B fp8 draft | n/a | 23.34 (W=1 tick) | 295.6 |
| 2026-09-06 | d1f2bb8 | H20 card 6 | cuda sm90 | 27B, spec d1 | n/a | 39.47 | 292.0 |
| 2026-09-06 | d1f2bb8 | H20 card 6 | cuda sm90 | 27B, spec d7 | n/a | 66.82 | 252.1 |

Raw artifacts: `/work/sweep48.log`, `/work/w1.log` on the pod.

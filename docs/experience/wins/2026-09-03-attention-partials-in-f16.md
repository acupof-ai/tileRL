# Attention partials in f16 — parity passed, and the peak was twice what OOMed, V100 sm70, 2026-09-03

> Status: **shipped on a passed parity gate** (`70b53f2`). `PO [B, S, H, KVSPLIT, D]` is now
> f16, halving the only allocation in the split-KV path that scales with the tick's padded
> width. **All 21 (n, S) pairs pass against the dense paged kernel, worst
> |split − generic| = 3.424e-04 — 30× inside `rtol=1e-2`**, and the same order as the
> 2.1-2.6e-4 predicted before the run. ctx=32 is **60.9 vs 61.1 tok/s, −0.4%, inside the
> 1.16% noise floor** — as predicted, a shape that already fits only moves bytes.
> **ctx=512 still OOMs, and f16 alone cannot open it.** Two arms at different headroom show
> the failing row tracks free memory rather than the peak (396 MiB free → dies at 5×512;
> 1.11 GiB free → dies at 7×512), so the peak is **8×512 = 1.500 GiB in f16** against 1.11 GiB
> free. **KVSPLIT 32→16 takes it to 0.750 GiB, which fits with 360 MiB spare** — and that flip
> is now blocked on its own measurement, because 16's recorded speed win is at prefill width
> only.

## Context

[The draft readout fix](2026-09-03-draft-readout-reduced-to-the-row-it-reads.md) removed
1.53 GiB and B=8 ctx=512 moved to a second wall: `paged_attention_split`'s partials at
1.50 GiB. Fragmentation was ruled out by experiment — `expandable_segments:True` took
reserved-but-unallocated 2.82 GiB → 261 MiB and free 396 MiB → 1.38 GiB, and the request
still failed, **123 MiB short**.

A research workflow mapped the footprint read-only across four lanes (KVSPLIT plumbing,
the shared-width padding, partials dtype/lifetime, and what gates a change), then had every
proposal adversarially verified against the code. Four survived. This is the one with the
largest margin, the smallest diff, and a verifier that **reproduced the numerics itself**
rather than estimating them.

## The change

`kernels.py`, 16 insertions / 7 deletions, one file:

- `PO = T.empty((B, S, H, KVSPLIT, D), "float16")` (was f32), store wrapped
  `T.cast(acc[d], "float16")`.
- The combine's input annotation becomes f16 and its read is widened:
  `o[d] += w * T.cast(PO[...], "float32")`. The merge arithmetic stays f32.
- **PM/PL stay f32.** They are 12 MiB together, and they hold the log-domain running max
  and logsum — that is where the range lives.
- Two docstrings corrected: PO's dtype, and the combine's stated arithmetic.

Why it is safe, not merely small: PO holds `exp(s − m) · V` with the running max **already
subtracted**, so the values are O(1). A torch simulation of the kernel's own arithmetic
measured **max|PO| = 9.33 against f16's 65504** (~3.8 orders of headroom) and rel err
2.1e-4 / 2.6e-4 at ctx=128 / 512, against f32's 2.5e-7.

An earlier worry was wrong and worth recording: **the partials are not under the f32-IO
constraint.** That constraint is `Backend.io` (`backend.py:170`), scoped to the fp4 GEMV
path; PO/PM/PL are kernel-internal between `backend.py:702` and `:713` with no dtype
coercion in between.

## Parity

`scripts/check_split_attn_parity.py` on the pod, split-KV against the **dense** paged
kernel (not f16 against f16):

| n | S=1 | S=2 | S=4 |
|---:|---:|---:|---:|
| 17 | — | 3.424e-04 | 3.087e-04 |
| 37 | 1.917e-04 | 2.112e-04 | 1.957e-04 |
| 100 | 1.178e-04 | 1.258e-04 | 2.570e-04 |
| 129 | 8.503e-05 | 1.363e-04 | 1.342e-04 |

All 21 pairs `OK`, `PARITY_EXIT=0`. Worst 3.424e-04.

**This gate cannot run on the Mac**, and that is not a consequence of this change:
`make_paged_attention_split` does not compile for `target="c"` — `tl.reduce requires a
target-specific implementation, but no reduce implementation is registered for cpu`.
**Negative control run: pristine HEAD fails identically.** So the split kernel has no CPU
twin today, before this diff. The three `T.reduce` calls at `kernels.py:702/708/712` are the
blocker, and they are also the sm70 thread-redundancy win, so a CPU twin needs its own
serial cell rather than a tweak — a standing gap, noted and not widened.

Two more gate facts found by the workflow: `check_split_attn_parity.py:29` pins
**KVSPLIT = 16** while `registry.py` ships **32**, so the correctness gate has never
exercised the shipped value; and its sequence lengths were chosen to straddle 16, so at 32
the ragged/empty-slice coverage sits off the slice boundary.

## Why f16 alone does not open ctx=512

A verifier transcribed `_build_plan` verbatim and drove it: the 1.50 GiB request that failed
is **4 rows × 512**, but the same run reaches **8 rows × 512 = 3.000 GiB**.

| | KVSPLIT=32 | KVSPLIT=16 |
|---|---:|---:|
| f32 | 3.000 GiB | 1.500 |
| **f16** | **1.500** | **0.750 ✓** |

Free after `expandable_segments`: **1.38 GiB**.

Then the run settled it, and it took two arms because the first one misled me. The f16 arm
OOMed asking for **960 MiB with 396 MiB free**. 960 MiB is not a round number — it decodes to
a shape: `960 MiB / (H·KVSPLIT·D·2 B) = 960·2^20 / (24·32·256·2) = 2560 = 5 × 512`. I read
that as "the peak is 5 rows", which was wrong.

The `expandable_segments` arm proved it wrong. Free rose **396 MiB → 1.11 GiB**, and the tick
did not succeed — it **died one row later**, asking **1.31 GiB = 7 × 512**, 207 MiB short:

| arm | free | died asking | = rows × 512 |
|---|---:|---:|---:|
| no flag | 396 MiB | 960 MiB | **5** |
| `expandable_segments` | 1.11 GiB | 1.3125 GiB | **7** |

**The row it dies on is a function of free memory, not of the peak.** Each row's partials are
allocated per-call and freed, so the tick walks up the row count until one request exceeds
what is free — give it more headroom and it gets further before failing. So the failing
request never named the peak, and the verifier's derivation of **8 rows** stands.

That makes f16 necessary and not sufficient, with the arithmetic now unambiguous:

| 8 rows × 512 | KVSPLIT=32 | KVSPLIT=16 |
|---|---:|---:|
| f32 | 3.000 GiB | 1.500 |
| **f16** | **1.500** | **0.750** |

Against 1.11 GiB free, **f16 + KVSPLIT=16 = 0.750 GiB fits with 360 MiB spare**, and f16 alone
(1.500) does not. Both halvings are needed.

Recording the estimate error honestly: I named **three** row counts for this peak (4 measured,
8 derived, 5 inferred from a byte count) and published the third. Dividing a failed
allocation by its per-row bytes gives a real shape — but **the shape of the request that
happened to fail is not the shape of the peak**, and one arm with more headroom was enough to
show it.

## KVSPLIT=16 is not yet a free win — the recorded number is at the wrong width

The workflow refuted the plan I was about to run. The **1.17× faster at KVSPLIT=16** figure
(4002 vs 4700 µs) is at **ctx=4096, S=32** — prefill width. A spec tick runs **S=1** (decode)
and **S=4** (verify at depth 3), and at S=1 the recorded case *for* 32 is real: a block owns
≤128 positions, below launch overhead, which is what flattened the context slope
(512→4096 went 157→163 µs).

Worse, the 16→32 flip arrived **bundled** with the thread-redundancy rewrite, so the split
count's own contribution was never isolated on the current kernel. So "16 is faster and
smaller, take it" is unsupported at the widths that matter. `scripts/prof_attn_ctx.py` swept
S=32 only; it now sweeps **S=1, 4, 32** and prints PO bytes per KVSPLIT, and that measurement
decides the flip.

One safety fact that removes a whole class of worry: a split/combine KVSPLIT mismatch
**raises at call time** and can never silently produce wrong numbers. KVSPLIT is baked as an
IntImm into the combine's input declarations (`kernels.py:739-741`), so the packed ABI asserts
on it — verified on both execution paths.

## What the padding actually costs

The footprint traces to `engine.py:724-729`: `rows = decodes + prefills`, then one bucketed
`width` for every row. Measured locally with `scripts/probe_mixed_tick_padding.py` on the
CPU target:

| tick kind | ticks | mean rows | mean width | useful | waste |
|---|---:|---:|---:|---:|---:|
| decode | 25 | 1.4 | 1.0 | **100%** | 0% |
| prefill | 1 | 1.0 | 64.0 | 31% | **69%** |
| mixed | 7 | 2.0 | 64.0 | 28% | **72%** |

Decode ticks waste nothing; the waste is entirely in ticks carrying a prefill chunk, and a
pure prefill tick is as bad as a mixed one. The sharper statement, verified at
`engine.py:563` and `:580`: **`max_num_batched_tokens` bounds only the summed chunk length,
never `rows × width`** — the footprint scales with a product the scheduler never accounts
for.

And it is a **compute** waste too, not only footprint: `kernels.py:684-685` states that a
padded query row still runs, so a decode row in a 512-wide tick walks a real 512-token row's
key visits; the same padding inflates every fp4 GEMV in the tick, since `backend.py:450`
flattens to `M = rows × width`.

## Rule

**When a local gate cannot run, run the negative control on the gate itself.** The parity
script failed on this Mac and the honest question was whether my diff caused it. Restoring
`kernels.py` from `git show HEAD:` and re-running showed pristine code failing identically,
which converted "my change broke the CPU path" into "this kernel has no CPU path" — a
different and more useful fact, and one that belongs in the tree rather than in a debugging
session.

Second: **a research pass earns its cost when it refutes the brief, not when it agrees.**
Three of the four items that changed what shipped were corrections: the peak is bigger than
the request that failed, the f32-IO constraint does not reach the partials, and the parity
gate has never run at the shipped KVSPLIT. None was in my framing when the workflow launched
— and its sharpest contribution was killing the *next* step I had queued, by pointing out
that 16's speed advantage is recorded only at prefill width.

Third: **a failed allocation names a shape, not the peak.** `960 MiB / (H·KVSPLIT·D·2)` = 5
rows exactly, which felt like a measurement and was not one — with more headroom the same run
died at 7 rows instead. Because these partials are allocated and freed per call, the tick
climbs until one request exceeds what is free, so the failing size measures the *headroom*,
and only a second arm at different headroom separates the two.

## Gate

Parity on the pod, 21/21, worst 3.424e-04 against the dense kernel. 191 tests pass, ruff
clean. Negative control on the CPU failure verified against pristine HEAD. GPU verified idle
before launch.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **PO f16 parity vs dense kernel** | **21/21 OK, worst 3.424e-04** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | predicted rel err before the run | 2.1-2.6e-4 — **same order** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | max\|PO\| vs f16 range | 9.33 vs 65504 (~3.8 orders) |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **PO at the failing shape (4×512)** | **1.500 → 0.750 GiB** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **the ctx=512 PEAK (8×512)** | **3.000 → 1.500 GiB, free 1.38** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | f16 + KVSPLIT=16 peak | **0.750 GiB — the pair that fits** |
| 2026-09-03 | (recorded) | V100 | cuda sm70 | attention | KVSPLIT 16 vs 32 @ctx4096 S=32 | 4002 vs 4700 µs — **16 is 1.17× faster** |
| 2026-09-03 | 70b53f2 | Mac | cpu | — | split kernel on target="c" | **does not compile — no CPU twin, predates this** |
| 2026-09-03 | (this) | Mac | cpu | tiny | mixed-tick padding | **72% waste; decode ticks 0%** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=32 with PO f16** | **60.9 tok/s vs 61.1 — −0.4%, inside noise** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=512, no expandable_segments** | **OOM at 960 MiB = 5×512 f16, free 396 MiB** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=512, expandable_segments ON** | **OOM at 1.3125 GiB = 7×512 f16, free 1.11 GiB** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | what those two arms prove | **the failing row tracks FREE memory, not the peak** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | peak (8×512) f16, KVSPLIT=32 | 1.500 GiB vs 1.11 free — **still short** |
| 2026-09-03 | (next) | V100 | cuda sm70 | qwen38-27b | peak (8×512) f16, KVSPLIT=16 | **0.750 GiB — fits, 360 MiB spare** |

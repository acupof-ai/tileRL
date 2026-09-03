# The sm70 attention partials: f16 plus a per-width split count opens B=8 at ctx=512, V100 sm70, 2026-09-03

> Status: **shipped, gate passed, the ceiling is gone** (`70b53f2` + `a158672`). Two changes:
> `PO [B, S, H, KVSPLIT, D]` is f16 (was f32), and the sm70 split count is now chosen per tick
> by query width instead of being one hard-wired 32. **B=8 ctx=512 runs at 88.5 tok/s where
> every earlier arm OOMed**, and ctx=32 is unchanged at 61.1 (baseline 61.1). Parity passes on
> the card at **both** shipped split counts.
>
> **88.5 vs 61.1 is not a speedup.** The ctx=512 tick is **1.063× slower** (244.1 vs 229.6 ms);
> the higher rate is entirely acceptance — **tok/forward 21.60 vs 14.00, 1.543×** — and those
> two multiplied give 1.451× against a measured 1.448×, **0.2% apart**. The result is that this
> configuration exists at all.
>
> **Caveat, now measured:** the 88.5 run carried
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the tree sets nowhere, and the arm
> without it **OOMs** — in the MLP, not attention.
> [errors/2026-09-03-expandable-segments-is-load-bearing.md](../errors/2026-09-03-expandable-segments-is-load-bearing.md)

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

## Why f16 alone did not open ctx=512

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
(1.500) does not. Both halvings are needed — and the run confirmed it: with the per-width
split count in, **ctx=512 runs**.

Recording the estimate error honestly: I named **three** row counts for this peak (4 measured,
8 derived, 5 inferred from a byte count) and published the third. Dividing a failed
allocation by its per-row bytes gives a real shape — but **the shape of the request that
happened to fail is not the shape of the peak**, and one arm with more headroom was enough to
show it.

## KVSPLIT=16 is not a free win — and the sweep pointed at a better change

The workflow refuted the plan I was about to run, and then the measurement refuted the
plan's premise. The **1.17× faster at KVSPLIT=16** figure (4002 vs 4700 µs) is at
**ctx=4096, S=32** — prefill width. A spec tick runs **S=1** (decode) and **S=4** (verify at
depth 3), and the 16→32 flip arrived **bundled** with the thread-redundancy rewrite, so the
split count was never isolated on the current kernel.

Swept at the widths that actually run (ctx=4096, block_N=16, µs/call):

| S | ks16 | ks32 | ks64 | 16 vs 32 | PO at 8×512 |
|---|---:|---:|---:|---:|---:|
| **1** (decode) | 246.5 | **205.1** | 192.5 | **0.832× — 16 loses 20%** | 3 MiB |
| **4** (verify d3) | 658.4 | 660.3 | 680.7 | 1.003× wash | 12 MiB |
| **32** (prefill) | 4660.2 | 4682.3 | 4850.3 | 1.005× wash | **1.500 GiB** |

**A flat flip to 16 would have cost 20% on decode** — the shape a spec tick is mostly made
of. So the flip I had queued as "free" was not free.

But the two constraints turn out to sit at **opposite ends of S**, and never conflict: 32
earns its 20% exactly where PO is 3 MiB and saving bytes is pointless, and 16 is free exactly
where PO is 1.5 GiB and OOMs the card. So **choose the split count by query width**, which is
not a new mechanism — `_paged_attention_decode` (`backend.py:762-764`) already derives sm90's
split count host-statically from shape, graph-safe. The sm70 path now does the same: `S < 8`
→ 32, `S ≥ 8` → 16, with the threshold above the widest verify the ladder can submit (depth 7
is S=8). A 512-wide tick's PO becomes **0.750 GiB against 1.11 GiB free**.

Two wiring facts this needed. The registry's closures **swallowed** the call site's choice —
`_SM70_KERNELS["paged_attention_split"]("c", KVSPLIT=8)` raised `TypeError`, so they are now
bare factories and `Backend._kernel` keys its compile cache on the argument, as it already
does for `linear_fp4_gemv_sm70_m`. And a split/combine mismatch **raises at call time** rather
than computing silently: KVSPLIT is baked as an IntImm into the combine's input declarations
(`kernels.py:739-741`), so the packed ABI asserts on it — verified on both execution paths.

## The result, and what 88.5 is not

With both changes in, B=8 depth 3 on the real 27B:

| ctx | tok/s | ms/tok | tok/forward | tick ms |
|---:|---:|---:|---:|---:|
| 32 | 61.1 | 16.4 | 14.00 | 229.6 |
| **512** | **88.5** | 11.3 | 21.60 | 244.1 |

**88.5 vs 61.1 is not a 1.45× speedup, and reporting it as one would be wrong.** The ctx=512
**tick is 1.063× slower** (244.1 vs 229.6 ms) — longer histories cost more attention, as they
should. The entire rate gain is **acceptance**: `tok/forward` rises **14.00 → 21.60, 1.543×**,
because a 512-token context gives the draft head far more to condition on than 32 tokens do.

Those two are a closed check on the measurement rather than two loose observations:
`1.543 / 1.063 = 1.451×` predicted, **1.448× measured — 0.2% apart**. Nothing else is needed
to explain the number, which is also why no part of it can be claimed for the kernel work.

**What the kernel work bought is that the row exists.** Every earlier arm at this
configuration OOMed: 1.41 GiB in the draft readout, then 1.50 GiB in the partials, then
960 MiB, then 1.31 GiB. ctx=32 confirms the price was zero where it should be — **61.1 against
a 61.1 baseline**, and by construction, since every tick there is S≤4 and takes the same
KVSPLIT=32 kernel it always took.

**Precondition, measured and confirmed:** this run carried
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is set **nowhere in the tree** —
`src/`, `packages/` and every script are free of it; it lived only in the pod runner. **The arm
without it OOMs**, so the flag is load-bearing and the 88.5 is *"reachable with the allocator
flag"*, not what the shipped server does. The failing allocation is no longer in attention: it
is the fused `gate_up` projection's f32 buffer at `backend.py:500`, **476 MiB = 7 rows × 512** —
the same padded shape, one buffer over.
[errors/2026-09-03-expandable-segments-is-load-bearing.md](../errors/2026-09-03-expandable-segments-is-load-bearing.md)

## One number does not reproduce

**I am naming the instrument rather than the result.** The recorded S=32 pair was
ks16 4002 / ks32 4700; this run reads **4660 / 4682**. ks32 reproduces to 0.4%, ks16 is **16%
slower than recorded**. The one change between the runs is PO f16, so either f16 costs ks16
16% at S=32 or one of the two runs is off. Unresolved. The **within-run** comparisons the
design rests on are unaffected — same process, same dtype, same call.

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

Fourth: **a knob measured at one width is not a knob you understand.** The queued flip to
KVSPLIT=16 looked free on a recorded 1.17×, and at the width a decode tick actually runs it
**loses 20%**. Sweeping the shipped widths did not merely kill the flip — it showed the
footprint constraint and the speed constraint sit at opposite ends of S, which is what made a
per-width choice the right change rather than a compromise between the two.

Fifth: **a rate that rises is not a kernel that got faster.** ctx=512's 88.5 tok/s against
ctx=32's 61.1 looks like a 1.45× win and the tick is actually **1.063× slower** — the gain is
acceptance, and multiplying the two closes to 0.2%. Splitting tok/s into tick time × tok/forward
before reporting it is what keeps a capacity fix from being written up as a speed fix.

## Gate

Parity on the pod at **both shipped split counts, 32 and 16** — `parity OK (cuda)`,
`PARITY_EXIT=0`; the f16 arm before it was 21/21 with worst 3.424e-04 against the dense
kernel. 192 tests pass, ruff clean. The width gate carries its own test with **two negative
controls verified**: lowering the threshold to 2 (a depth-3 verify would take the slow kernel)
fails it, and dropping `KVSPLIT` from the combine call — the real ABI-mismatch bug — fails it.
Negative control on the CPU failure verified against pristine HEAD. GPU verified idle before
each launch.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **PO f16 parity vs dense kernel** | **21/21 OK, worst 3.424e-04** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | predicted rel err before the run | 2.1-2.6e-4 — **same order** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | max\|PO\| vs f16 range | 9.33 vs 65504 (~3.8 orders) |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **PO at the failing shape (4×512)** | **1.500 → 0.750 GiB** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **the ctx=512 PEAK (8×512)** | **3.000 → 1.500 GiB, free 1.38** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | f16 + KVSPLIT=16 peak | **0.750 GiB — the pair that fits** |
| 2026-09-03 | (recorded) | V100 | cuda sm70 | attention | KVSPLIT 16 vs 32 @ctx4096 **S=1** | **246.5 vs 205.1 µs — 16 LOSES 20%** |
| 2026-09-03 | (recorded) | V100 | cuda sm70 | attention | KVSPLIT 16 vs 32 @ctx4096 **S=4** | 658.4 vs 660.3 µs — 1.003× wash |
| 2026-09-03 | (recorded) | V100 | cuda sm70 | attention | KVSPLIT 16 vs 32 @ctx4096 **S=32** | 4660.2 vs 4682.3 µs — 1.005× wash |
| 2026-09-03 | (recorded) | V100 | cuda sm70 | attention | the recorded 4002 for ks16 | **does NOT reproduce — 4660, 16% slower; ks32 reproduces to 0.4%** |
| 2026-09-03 | (next) | V100 | cuda sm70 | qwen38-27b | **per-width KVSPLIT parity, both counts** | **parity OK (cuda) at 32 and 16, PARITY_EXIT=0** |
| 2026-09-03 | a158672 | V100 | cuda sm70 | qwen38-27b | ks=16 worst error vs dense | **1.493e-04 — tighter than ks=32's 3.424e-04** |
| 2026-09-03 | a158672 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=32, per-width KVSPLIT** | **61.1-61.2 tok/s vs 61.1 baseline — flat, as predicted** |
| 2026-09-03 | a158672 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=512, per-width KVSPLIT** | **88.5 tok/s — RUNS; was OOM at every earlier arm** |
| 2026-09-03 | a158672 | V100 | cuda sm70 | qwen38-27b | ctx=512 tick vs ctx=32 tick | 244.1 vs 229.6 ms — **1.063× SLOWER per tick** |
| 2026-09-03 | a158672 | V100 | cuda sm70 | qwen38-27b | ctx=512 tok/forward | **21.60 vs 14.00 — 1.543× more accepted** |
| 2026-09-03 | a158672 | V100 | cuda sm70 | qwen38-27b | those two multiplied vs measured tok/s | **1.451× predicted, 1.448× measured — 0.2% apart** |
| 2026-09-03 | 70b53f2 | Mac | cpu | — | split kernel on target="c" | **does not compile — no CPU twin, predates this** |
| 2026-09-03 | (this) | Mac | cpu | tiny | mixed-tick padding | **72% waste; decode ticks 0%** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=32 with PO f16** | **60.9 tok/s vs 61.1 — −0.4%, inside noise** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=512, no expandable_segments** | **OOM at 960 MiB = 5×512 f16, free 396 MiB** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | **B=8 ctx=512, expandable_segments ON** | **OOM at 1.3125 GiB = 7×512 f16, free 1.11 GiB** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | what those two arms prove | **the failing row tracks FREE memory, not the peak** |
| 2026-09-03 | 70b53f2 | V100 | cuda sm70 | qwen38-27b | peak (8×512) f16, KVSPLIT=32 | 1.500 GiB vs 1.11 free — **still short** |
| 2026-09-03 | (next) | V100 | cuda sm70 | qwen38-27b | peak (8×512) f16, KVSPLIT=16 | **0.750 GiB — fits, 360 MiB spare** |

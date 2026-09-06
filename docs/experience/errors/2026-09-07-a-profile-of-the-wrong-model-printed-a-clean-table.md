# A profile of the wrong model printed a clean table — 2026-09-07

> Status: Fixed in the instrument. Every per-op share quoted for sm70 prefill on
> 2026-09-07 is withdrawn — CPU and V100 alike. The mechanism survives.

## Context

#213 established that V100 prefill is per token and quadratic. #215 shipped
`scripts/prof_prefill_ops.py` to attribute the n² to a kernel. The V100 arm ran
in a maintenance window and produced a table: `paged_attention` 32.6% at 16k,
`linear_attn_chunk` 35.4%, a per-chunk fit at R² 0.9999, and a slope 125x below
the CPU twin's.

All of it describes **`tiny`** — 2 layers, hidden 64, random weights — not the
27B. The shares are void.

## Root Cause

Two mistakes, and they are different.

**Mine, the operational one.** I handed over a command containing
`--model qwen38-27b` that the script itself refuses (`main` raises `SystemExit`
on any model but tiny — that guard was deliberate, for the CPU gate), and two
absolute paths, `/work/tl013/bin/python` and `/work/prefill_ops.json`, that
belong to the H20 pod. There is no `/work` mount on the V100 at all. The peer
checked both before running rather than guessing, which is the only reason the
window was not spent on a run that would have failed after the card went idle.

**Mine, the analytical one.** I offered the CPU profile's 64% attention share as
a prior for the 27B. It is not one, and the gap is not small. Per token,
attention's prefix scan is `full_attn_layers · H · D · 4` FLOP and the linears
are `≈ L · 2 · (4·hid² + 3·hid·inter)`:

| | layers | full-attn | H | D | hidden | inter | attn==linear at prefix |
|---|---|---|---|---|---|---|---|
| tiny-agent | 2 | 1 | 4 | 16 | 64 | 128 | **640** |
| qwen38-27b | 64 | 16 | 24 | 256 | 5120 | 17408 | **121,173** |

At n=16,384 the attention/linear FLOP ratio is **12.8 for tiny and 0.07 for the
27B** — a factor of ~190, in the dimension the profile was measuring. Tiny is a
model where attention dominates almost immediately; the 27B is the opposite at
every length that matters here.

## Why nothing caught it

**Every assertion asked whether the instrument worked. None asked whether the
number was the right size.** The selfcheck verified that each tick timed at least
one op, that the first chunk's prefix was 0, that chunks summed to the prompt,
and that attention was on the timed path. All four are true of a profile of the
wrong model, and all four were true here.

The evidence was in the output the whole time: the run totalled **0.813 s at
16,384 tokens** where #213's own fit predicts **179.7 s** — **221x**. Two numbers
from the same investigation, in the same session, never divided by one another.

A second void arm passed the same gate. The 2048-token arm's first chunk read
`paged_attention` 5.276 s and `write_tokens` 4.008 s against 0.001–0.002 s for
every later chunk: TileLang compiling `paged_attention_split`, `write_tokens_f32`
and `gdn_prep` **inside the timed region**. It would have reported a 42.3%
attention share that was a compile. First-prefix-is-0, chunks-sum and ops-timed
are all true of a compile too.

## Fix

Three gates, each aimed at one of the ways a clean table can be wrong.

**Reconciliation against the fit.** The per-chunk sum must land within 2x of
`C1·n + C2·n²` from #213, asserted, with the ratio printed. Only the n and n²
terms: the per-chunk sum has no decode step and no route overhead, which is what
the fit's constant absorbs. 2x, because the sync overhead alone was 36% at 16k
and the fit carries its own error — the failure this catches was 221x. Runs that
are not the 27B on sm70 print an explicit banner instead:
`SHARES HERE DO NOT DESCRIBE THE 27B`.

**A warm-up arm at the real shapes, outside the timed region**, with its own
printed line so a compile cannot be re-read as a chunk. At the *largest* arm's
shapes: TileLang keys its cache on tile shapes, so warming at 2k compiles nothing
the 16k arm uses.

**A chunk-0 outlier assertion** — the first chunk may not exceed `max(20 × median
of the rest, median + 0.05 s)`.

The script now loads the served checkpoint through `_build_model`, the same entry
`serve` and `prof_grpo_step.py` use, so the sm70 path measures the 27B's shapes.

## Gates

Both new gates were exercised on the V100's **actual** numbers, not on synthetic
stand-ins:

| input | result |
|---|---|
| the real compile arm, `[5.276, 0.10, 0.16, 0.23]` | fails: `chunk 0 took 5.276 s against a median 0.160 s` |
| a healthy rising series | passes |
| the real tiny run, 0.813 s at n=16,384 | fails: `0.00453x, tolerance 0.50-2.00x` |
| a plausible 27B run, 180 s at n=16,384 | passes (1.00x) |

**The warm-up gate is unobservable on cpu** — removing it still passes there,
because the CPU target has no JIT to warm — so the chunk-0 assertion is the gate
that actually runs in CI, and it is the one exercised against the recorded
numbers above. No GPU CI arm exists to close that, and none is added for this.

## What survives

The mechanism, and it survives on real sm70 kernels. Only `paged_attention` rises
with prefix — **0.0006 → 0.0138 s over prefix 0 → 15,872, 22.86x** — while every
other op stays flat within 9%, at R² 0.9999. The provenance matrix confirms from
the compile log that `paged_attention_split` is the kernel that ran, so the
CPU-authored `make_paged_attention` is registered on sm70 and unreachable.

**The quadratic is in attention. Its share of the 27B's prefill is unmeasured.**

## Rule

An instrument that validates itself validates nothing about the measurement. Every
profile needs one assertion that compares its output to an independent estimate of
the same quantity — and the estimate usually already exists in the same
investigation. Here it was one division away for three hours.

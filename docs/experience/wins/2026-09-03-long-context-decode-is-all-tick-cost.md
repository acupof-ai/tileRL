# Long-context decode at B=1, on a prompt that only varies in length — sm70, 2026-09-03

> Status: Shipped (measurement; supersedes the withdrawn rows in
> errors/2026-09-03-the-context-sweep-changed-the-prompt.md)

## Context

The fitted KV pool opened contexts past 4096 for the first time, and the first
sweep there was confounded: the prompt was `range(10 + i*ctx, ...)`, so each
length read a different slice of the vocabulary and `tok/forward` mixed two
causes. Those rows are withdrawn. This is the re-measurement with a prompt drawn
from one fixed distribution, so length is the only variable — three points, one
process, one pool.

## What Worked

> **Label added 2026-09-04:** these rows are `--tokens 64`, recoverable only by
> inverting the 554-block pool line. `--tokens` IS the measurement window, so
> tok/forward here is not comparable to any 128-token row — the same code reads 2.44
> at ctx=1024 with `--tokens 128`. The rows are correct; they were unlabelled, and
> four commits were investigated for the resulting gap.
> [errors/2026-09-04-four-candidates-cleared-for-a-flag-difference.md](../errors/2026-09-04-four-candidates-cleared-for-a-flag-difference.md)

| ctx | tok/s | ms/tok | tok/fwd | tick ms |
|---:|---:|---:|---:|---:|
| 1024 | 38.8 | 25.8 | 2.03 | 52.3 |
| 2048 | 34.1 | 29.3 | 2.10 | 61.6 |
| 4096 | 31.8 | 31.4 | 2.10 | 66.0 |

`tick ms = ms/tok × tok/forward`, derived from the two measured columns.

**Acceptance is flat in context: 2.03 → 2.10, a 1.034x rise over 4x the
context.** The whole rate curve is tick cost — 1.262x tick against a 1.220x rate,
and `1.262 × (2.03/2.10) = 1.220` closes exactly. That is the opposite of what the
confounded sweep said, where acceptance appeared to carry 1.409x of a 1.80x drop.
(The attribution below splits that tick growth further: it is not all context.
`verify_lens` widens the chain as context rises, and the wider rung is most of it.)

The prompt change alone, at fixed ctx=1024, moves acceptance **2.86 → 2.03
(1.409x)** and the tick **62.3 → 52.3 ms (0.840x, narrower chains are cheaper)**,
netting the 1.183x rate difference against the old record's 45.9. So 45.9 tok/s was
not wrong as a measurement; it was measured on a prompt the draft finds unusually
easy.

## Rule

`tok/forward` on this stack is a property of the prompt distribution, not of the
context length. Quote it with the distribution it was measured on, and never
compare two acceptance numbers from different prompts. A random-vocabulary prompt
is the pessimistic end (2.03 at W=4, below the 2.776 break-even); consecutive low
ids are the optimistic end (2.86, just above). Neither is the serving distribution,
so **the depth default cannot be settled on either** — that needs real text.

**Settled 2026-09-04 on wikitext-103, and random vocabulary was NOT the pessimistic
end**: text accepts 2.36 at W=4 against random ids' 2.99, so the "pessimistic end"
label above is wrong. Depth 1 beats the shipped depth 3 by 1.266x on text and loses
on random ids. [wins/2026-09-04-depth-default-is-wrong-on-text.md](2026-09-04-depth-default-is-wrong-on-text.md)

## Open, and load-bearing for the next decision

Tick cost is **not linear in context**: 9.08 ms per 1K over 1024→2048, then 2.15
over 2048→4096. A byte-bound cost would be flat, and the 4-row f32 KV floor is
0.597 ms per 1K (128 KiB/token at 900 GB/s), so the first interval runs ~15x the
floor and the second ~3.6x. Something steps between 1024 and 2048 rather than
scaling. Three points cannot say what; `scripts/prof_decode_budget.py` attributes
by kernel inside the captured graph and can. Not guessed at here.

## 2026-09-03, later: answered — the step was a rung, and context cost IS linear

Profiled by kernel at 1024 and 2048. `fp4 GEMV` carried **+7.09 ms of the +10.79
ms/forward** at an *identical* call count (330.2 → 330.0 calls/forward), which no
shape change explains — until the chain width is printed. The mean drafted chain
rises **2.28 → 3.70**, crossing the `LADDER_WIDTHS` rung 2 → 4. So that GEMV
growth is the verify rung, not the context: the same staircase error as
`errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md`, this time hidden
because the launch count is per-layer and identical on both rungs.

The clean control is the **dense** path, which is always rung 1, so the GEMV
shapes cannot move:

| ctx | GPU ms/fwd | fp4 GEMV | attention | everything else |
|---:|---:|---:|---:|---:|
| 1024 | 24.92 | 19.34 | 1.15 | 4.43 |
| 2048 | 25.55 | 19.35 | 1.75 | 4.45 |
| 4096 | 26.73 | 19.34 | 2.95 | 4.44 |

**At a fixed rung, context costs attention and nothing else — 99.4% of the
1024→4096 delta — and it is linear to the last digit**: 0.600 ms per 1K over both
intervals, fitting `0.55 + 0.600 × ctx/1K` at 1.15 / 1.75 / 2.95 exactly. GEMV is
flat to 0.05% across 4x the context, which is also this arm's order control: 1024
ran first in every arm, and drift over a process would move GEMV too. (The spec
arm was order-controlled directly: 1024 read 29.43 running first and 29.50
running second, 0.24% apart.)

So this entry's "not linear" claim above is **withdrawn**. It came from reading a
rung crossing as a context cost.

Two numbers this leaves:

- **Attention runs 4.0x its own byte floor.** One dense row streams 128 KiB/token
  of f32 KV, so 1K of context is 0.149 ms at 900 GB/s against 0.600 measured. The
  0.597 ms/1K in the withdrawn paragraph was the *4-row* floor compared against a
  1-row-equivalent slope — a mismatch that made attention look ~15x off when the
  gap is 4.0x.
- **The wider chain at long context is a loss, not a win.** Going 2.28 → 3.70
  chain buys **+0.16 tok/forward** (1.88 → 2.04) and costs **+10.0 ms/forward**
  of GEMV and attention. `verify_lens` trims on the draft's confidences, which
  rise with context, so it spends a rung to keep tokens the verifier then
  rejects. Capping verify width by context is a candidate lever; the size of it
  is **not derived here**, because the profiler's wall (82.0 ms/fwd) carries
  profiling overhead the 61.6 ms bench tick does not, and dividing one by the
  other would be a fabricated ratio. It needs a direct A/B at ctx=2048 with the
  rung pinned.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-03 | 6345e1a | V100 32GB | cuda sm70 | qwen38-27b | — | 25.8 | 38.8 |
| 2026-09-03 | 6345e1a | V100 32GB | cuda sm70 | qwen38-27b | — | 29.3 | 34.1 |
| 2026-09-03 | 6345e1a | V100 32GB | cuda sm70 | qwen38-27b | — | 31.4 | 31.8 |

Single-stream, spec depth 3 (W=4), slots 3, one pool sized for ctx=4096, prompt
from `_prompt(ctx, 0, 248320)`. Rows are ctx 1024 / 2048 / 4096.

Raw artifacts: `$HOME/tilerl-logs/lcfix2.log` on the V100. The attribution pass is
`$HOME/tilerl-logs/bud4.log` (spec, rung crossing) and `bud5.log` (dense control),
from `prof_decode_budget.py --ctx 1024,2048[,4096]`, 48 tokens per arm. Those are
GPU-time-inside-the-profiler numbers and are not comparable to the wall-clock rows
above; they are only compared to each other.

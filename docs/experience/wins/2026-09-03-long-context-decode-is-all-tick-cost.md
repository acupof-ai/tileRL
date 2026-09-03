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

## Open, and load-bearing for the next decision

Tick cost is **not linear in context**: 9.08 ms per 1K over 1024→2048, then 2.15
over 2048→4096. A byte-bound cost would be flat, and the 4-row f32 KV floor is
0.597 ms per 1K (128 KiB/token at 900 GB/s), so the first interval runs ~15x the
floor and the second ~3.6x. Something steps between 1024 and 2048 rather than
scaling. Three points cannot say what; `scripts/prof_decode_budget.py` attributes
by kernel inside the captured graph and can. Not guessed at here.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-03 | 6345e1a | V100 32GB | cuda sm70 | qwen38-27b | — | 25.8 | 38.8 |
| 2026-09-03 | 6345e1a | V100 32GB | cuda sm70 | qwen38-27b | — | 29.3 | 34.1 |
| 2026-09-03 | 6345e1a | V100 32GB | cuda sm70 | qwen38-27b | — | 31.4 | 31.8 |

Single-stream, spec depth 3 (W=4), slots 3, one pool sized for ctx=4096, prompt
from `_prompt(ctx, 0, 248320)`. Rows are ctx 1024 / 2048 / 4096.

Raw artifacts: `$HOME/tilerl-logs/lcfix2.log` on the V100.

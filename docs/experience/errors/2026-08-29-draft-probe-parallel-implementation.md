# Five bugs in a row from a probe that re-implemented the engine — 2026-08-29

## Context

Measuring the bundled NextN/MTP draft head's acceptance rate against the 27B
trunk. `scripts/draft_probe.py` hand-built its own KV pool, state pool and
block loop instead of driving the engine. Five readings were reported over one
day; every one was void.

| reading | why it was void |
|---|---|
| 17.6% @ depth 4 vs 85.5% @ depth 8 | a fixed block count coupled draft depth to generation length — depth 8 simply generated further into the model's repetition regime |
| 93.9% / 98.3%, flat survival 0.79 / 0.88 | rejected drafts were never rolled back, so the probe scored the draft against text it wrote itself. A real survival curve decays; a flat one is the tell |
| 47.4% with the other fc order | same defect |
| degenerate continuation | no GDN recurrent-state rollback; the state compounds and has no self-healing path |
| 0.0% over 255 blocks | the draft's `BatchKv` carried the TRUNK's `LinearStatePool`; a 1-layer full-attn head wrote the trunk's recurrent state before the snapshot was taken |

## Root Cause

The probe was a parallel implementation of the engine's decode path. Each fix
made it converge one step closer to what the engine already does, and the next
divergence became the next bug. The self-check added after bug 2 ("committed
token must equal the trunk's argmax") was vacuous at 0% acceptance: with
`n_ok = 0` the committed token IS that argmax by construction, so it could
never fail.

The engine was never at fault. `scripts/parity_chunk_vs_decode.py` proved it in
two runs: a mid-sequence multi-token forward matches T=1 greedy exactly (chunk
widths 2/5/9, 15/15 each), and the block loop reproduces greedy exactly with a
draft that is always wrong (32/32) or always right (32/32).

## Fix

Speculation lives in the engine; the probe is deleted. Acceptance is read off
engine counters, so the thing measured and the thing shipped are one code path.

## Rule

A measurement tool that re-implements the system under test measures the tool.
Before trusting any number from a new probe, drive it to a case whose answer is
known independently — and check that the case can actually fail. A self-check
that holds by construction in the regime you are in is not a check.

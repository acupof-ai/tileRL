# I called a measured rejection "an argument" — because I only found the argument

## Context

`docs/experience/wins/2026-08-24-gdn-prefill-chunk.md` contains a one-line
aside dismissing fla's chunk delta rule as "incompatible with decay-first".
That line is wrong on the mathematics: the intra-chunk term it says is dropped
is carried by `wy_fast`'s `W`/`U`, and the chunked form is the same recurrence
reassociated (proved to 3.7e-07 in `reference.gdn_chunk_core`).

On 2026-08-29 I wrote an entry here concluding that chunkwise-WY had therefore
been "ruled out on an argument, not a parity run", and that a 1.36x prefill win
had been closed off for four days.

## Root Cause

That conclusion was false, and this file is its retraction. The tree already
held TWO measured rejections that I did not look for:

- `2026-08-25-gdn-prefill-wy-rejected.md` — a 2-kernel WY port, 2.6x slower.
- `2026-08-25-gdn-chunked-gdr-rejected.md` — the full 6-kernel FlashQLA
  pipeline ported from agent-infer, A/B'd at the real prefill-512 shapes:
  serial 4.380 ms vs chunked **4.882 ms**, and `max|d|` 8.51 on the output at
  scale=1.0 inputs because bf16 intermediates flow between the six stages.

I searched the wins entry that made the claim and stopped there. Both
rejections are in the same directory, both dated four days earlier, both with
A/B tables.

## What Survives

The mathematics stands, and so does the distinction the old A/B actually
measured. What was rejected is **six kernels, bf16 intermediates between
stages, and 8x the block count** — not the chunked recurrence itself. Its two
named failure modes are specific and avoidable:

- precision: fuse the stages so intermediates stay f32 in registers/shared.
- parallelism: 48 value heads already saturate 78 SMs, so more blocks buy
  nothing; keep one launch and the same block count.

So the live question is narrow: does the chunked recurrence win **inside one
kernel, at f32, with the block count unchanged**? That is a different arm from
either rejection, and it has to clear a gate the old one failed — accuracy at
scale=1.0 inputs, not the scale=0.1 of the parity fixtures.

## Rule

Before writing "this was never measured", grep the whole experience directory
for the technique, not just the entry that dismissed it. A weak argument in one
file is not evidence that no one ran the experiment; here two A/B tables sat
one directory listing away. And when retracting someone's conclusion, state
which arm they measured — "chunked is slower" and "these six kernels in bf16
are slower" are different claims, and only the second was ever tested.

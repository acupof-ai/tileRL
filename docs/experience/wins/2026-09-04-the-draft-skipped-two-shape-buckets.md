# The draft skipped two shape buckets the trunk has, and a served first visit paid 15.7x for it

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), 27B NVFP4 + 456M draft, `serve --depth 1 --max-batch 1 --max-ctx 4096`
**Task:** #73
**Commits:** `029b27c` (width), `6c6f6df` (block table)
**Instrument:** `scripts/probe_served_rate.py`; compile counts from the server log, request windows stamped with `date` on both sides

## What was wrong

`engine.py:740` rounds the trunk's prefill width to `_PREFILL_BUCKET = 64`, and
`engine.py:666` sizes the trunk's block table to the whole pool, with the comment
"Table width = pool size: the kernels compile it in, so a per-tick width recompiles."

`spec.py` did neither. The draft took `w = max(hi - lo + 1)` and
`nb = max(len(r.blocks))`, both raw. The draft runs the same two kernels that bake
their shape — `write_tokens_f32` and `paged_attention_split`, the only two taking
`seq_q_lens` — so **every distinct prompt length compiled them inline**, and the block
table added a second axis growing one column per 16 tokens of context.

## Measured, before and after

Same probe, same server config, first visit to each prompt length:

| prompt | compiles | wall | end-to-end | arm |
|---:|---:|---:|---:|---|
| 13 | 0 | 1650 ms | 46.1 tok/s | before, S=13 already seen |
| 37 | **14** | **17320 ms** | **4.4 tok/s** | before, new length |
| 24 | **14** | 14766 ms | — | before, a second new length |
| 13 | 8 | 26367 ms | 1.5 tok/s | width bucketed only |
| 44 | 4 | 15974 ms | 2.5 tok/s | width bucketed only |
| 13 | **0** | 2203 ms | 18.2 tok/s | both fixes |
| 21 | **0** | 1109 ms | 36.1 tok/s | both fixes |
| 44 | **0** | 1104 ms | 36.2 tok/s | both fixes |
| 48 | **0** | 1104 ms | 36.2 tok/s | both fixes |

On a first visit to a new prompt length: **17320 → 1104 ms, 15.7x**, and the rate goes
**4.4 → 36.2 tok/s, 8.2x**. Four distinct lengths now compile **nothing** and the wall
is flat — 1104/1104/1109 ms for 21/44/48 tokens — where before every unseen length was a
14-compile cliff.

Matched at 76 completion tokens, the configuration the earlier rows used: **41.9 tok/s,
1815 ms, 1.90 tok/forward, acceptance 0.875** against the earlier 46.1 / 1650 / 1.95 /
0.974. Different prompt, so acceptance differs and the rates are not comparable; the
compile count is what this entry measures.

## How it was located, after four wrong guesses

The mechanism came from **reading `params.pkl` out of `~/.tilelang/cache`** for the
kernels written during a specific request, not from reasoning about the code:

```
before: [1, 37, 4, 256]  for a 37-token prompt     <- S = prompt_tokens
        [1, 24, 4, 256]  for a 24-token prompt
after width fix: [1, 64, 4, 256] with tables [1,1] [1,3] [1,4] [1,5] [1,6]
after both:      nothing compiled
```

The trunk's own tensor for the same tick is 64 wide, which is what made the draft the
only candidate. Four guesses preceded it and each was eliminated:

1. **`_PREFILL_BUCKET`** — 13/15/37 all round to 64, so prefill shape is identical across
   arms that differed 0 vs 14 compiles.
2. **Block count** — `13+76` grows 1→6 blocks and `37+76` grows 3→8; both add 5.
3. **A per-call cache key** — refuted by an identical repeat compiling zero times.
4. **"Each new request opens a new shape"** — refuted by 14 compile pairs arriving across
   requests whose prompts were byte-identical.

Guess 4 is the one worth keeping: it was *consistent with* the first data I had, and the
data that killed it (identical prompts, steady 9.0 s pair spacing) was already in the same
log I had read.

## Why the fix is safe

The padding rows never reach the pool: `kernels_mma.py:71` gates the write on
`if t < SeqQLens[b]`, and `sq` stays exact. `_PREFILL_BUCKET` moved to `spec.py` because
`engine` imports `spec`, and both paths have to round identically or the bug comes back on
one side. The verify path shares the same table via `bt[live]`, so one change covers both.

## The gate

`tests/test_e2e.py::test_the_draft_prefill_width_is_bucketed_like_the_trunks` asserts on
the widths **`draft.forward` actually sees** — three prompt lengths (19/37/53), all of
which must arrive as 64, and one distinct block-table width — rather than mirroring the
arithmetic. A formula mirror passes even when `spec.py` stops calling it, which is the
failure mode this bug had for its whole life.

Both mutations were run and both fail it: reverting the width rounding, and reverting
`nb` to `max(len(r.blocks))`. 256 passed / 8 skipped.

## What the widening cost, priced rather than assumed

`nb = self.kv.num_blocks` widens a per-tick H2D copy, and `spec.py` builds its table with
plain `torch.zeros` where `engine.py:667` pins its own. The draft's `step` runs on every
decode tick (124 against 5 prefills in one served request), so this is ~25 copies per
request, not one. Measured with `scripts/probe_bt_copy.py`, 200 reps, against the 5.54 ms
draft forward — both `nb` arms at both `n`, because comparing an n=8 row against the n=1
pre-fix row read `+101 us` of "widening cost" that was mostly eight rows instead of one:

| n | nb 6 → 256 | nb 6 → 4146 |
|---:|---:|---:|
| 1 | **−1 us** (−0.02%) | +12 us (0.22%) |
| 8 | +25 us (0.45%) | +135 us (2.44%) |

**At the shipped config (B=1, 256 blocks) the widening is free** — a 2 KiB copy is all
launch overhead, so 256 columns cost the same as 6. Memory is a non-issue at every width:
259 KiB worst case against 24.4 GiB of weights.

Pinning recovers 97 of the 135 us at B=8/ctx=8192 (0.254 → 0.157 ms). **Not done**: that
configuration OOMs on sm70 at any depth (#72), and 2.44% of one term of the tick is under
the harness's own 1.16% noise floor once expressed as a tick fraction. The number is here
so it can be re-read rather than re-derived when B=8 becomes reachable.

## Rule

When two code paths run the same shape-specialized kernel, a bucket on one of them is not
a property of the system — it is a property of that call site. `engine.py` had the comment
explaining exactly why the block table must be pool-sized, and `spec.py`, written later
against the same kernels, did not inherit it.

And a cache directory is a measurement. Four hypotheses about what varied were each
plausible and each wrong; the shapes were sitting in `params.pkl` the whole time, and
reading them took one command.

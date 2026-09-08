# The prediction for the 2.6x rollout-tick arm — written before the run

**Status:** prediction on record, arm not yet run. No card claimed at the time of writing.
**Written:** 2026-09-08, against `abb5604`.

This file exists so the arm's result can be read as a surprise or a confirmation. A
decomposition that surprises is worth more than one that confirms, and there is no way
to tell which one happened unless the expectation is on record first. The idiom is
`scripts/probe_norm_cast_once.py`'s docstring; the framing is `tilerl-27`'s.

## The quantity, and why not ms/tick

The defect is recorded as a gap in `ms/tick` — 55.21 and 61.68 training against 23.34
serving, all `wall / decode_forwards`
([the entry](../errors/2026-09-08-the-training-rollout-tick-is-2.6x-serving.md)). That
figure divides by a count which can itself differ between the arms, so a gap in it does
not say where the time goes. The arm measures the identity instead:

    wall/token = (forwards/token) x (device time/forward) + (residual/token)

Three terms, three different instruments:

| term | quantity | instrument |
|---|---|---|
| 1 | `decode_forwards / tokens_generated` | both counters already published (`engine.py:887,889`); the two increment sites (`:1026` eager, `:1229` graph) are mutually exclusive — a graph tick returns at `:977` before the eager one |
| 2 | device time per forward | CUDA events around the replay. **New instrumentation**: there is no `_decode_secs` — prefill has `_prefill_tokens`/`_prefill_secs` (`:482-483`) and decode has only a count |
| 3 | residual | `wall - Σ(term 2)`, named and asserted, never assumed small |

## The gate — one constraint, and why it is not enough on its own

**Closure.** The three terms must reconstruct each arm's own observed figure, not merely
differ between arms. If serving's terms do not rebuild 23.34 and training's do not
rebuild 61.68, there is a fourth term that has not been named and the comparison means
nothing.

**Closure cannot detect a wrong term 2, structurally.** Term 3 is *derived* as
`wall − Σ(device time)`, so `wall = Σ(device) + residual` holds by construction for any
value of term 2 whatsoever. The gate would go green over a CUDA-event measurement that
is wrong by 10x, because terms 2 and 3 absorb the error in equal and opposite amounts.
Closure tests term 1 against the total and nothing else. (Found by `tilerl-27`, who
issued the gate and then found the hole in it.)

**So term 2 needs a bound from outside the identity.** `_prefill_secs` (`engine.py:483`,
incremented `:1023`) supplies one, on a quantity the engine already publishes, in the
same process. Instrument the prefill path with the same CUDA events and require:

    Σ(prefill device events) ≤ _prefill_secs        and a plausible fraction of it

Three facts about that anchor, read rather than assumed:

- **It is wall, not device**: `time.perf_counter()` at `:1001` and `:1023`. So it is an
  upper bound, not an equality — and the difference between the two *is* host overhead,
  which is the thing being measured elsewhere. Treating it as an equality would assume
  away the answer.
- **On a prefill-only tick nothing in the interval forces completion**, so the bound does
  not hold as written. `_sample_commit` (`:1018`) sits in the `else` of `if chains`, not
  under `if decodes` — on a prefill-only tick it *is* called, with an empty list, and
  `_sample_batch` returns `[]` at `:1309-1310` before reaching any `.tolist()`. No sync.
  `_finish_prefills` is at `:1024`, after the accumulation. The comment at `:1021` says
  both mixed and prefill-only ticks accumulate here, so the interval can close with
  kernels still in flight and `_prefill_secs` can be **smaller** than the device time it
  contains. `Σ(events) ≤ _prefill_secs` then fails with a correct instrument — in the
  direction that looks like a finding.
- **The fix is an explicit `torch.cuda.synchronize()` in the instrumented build only**,
  immediately before the interval closes, so the bound holds at any tick composition. The
  alternative — anchor only on mixed ticks, where `_sample_commit` has a non-empty list
  and does sync — is correct but makes the anchor's validity depend on tick composition,
  a precondition someone will forget.

The general rule: **a wall-clock interval bounds device time only if something forces
completion before it closes.** Otherwise it is not a bound in either direction, and the
check is worse than none. (Gap found by `tilerl-27`; the empty-list path is mine, after
their reading located the region.)

All line numbers here are against `b4e9360`. They shift by 5 under
[#290](https://github.com/acupof-ai/tileRL/pull/290), which deletes an unreachable
method above them — the first draft of this file cited the post-#290 numbers, read out
of that branch's worktree.

A ratio near 1% or 99% means the events are wrong before they are ever pointed at
decode. This check can come back false; closure cannot.

## The prediction

Term 1 is already partly measured and it is **not** where the gap lives:

| quantity | value |
|---|---:|
| observed gap (61.68 / 23.34) | 2.643x |
| term 1, if `tok/fwd` 6.90 vs a nominal 8 | 1.159x |
| remainder for terms 2 and 3 | **2.279x** |

So term 1 accounts for 9.7% of the excess and 2.279x has to be hiding in per-forward
device time or in the residual. Both are large places for it to hide.

**What I expect:** the residual carries the majority. The reasoning is that term 2 is
the same captured graph replaying the same shapes in both arms — the training engine
builds `decode_graph=True` and the measurement is graph-on both sides — so a 2.3x
difference in device time per replay would need a different kernel or a different shape,
and the arms agree on both. That leaves the host side.

**What would refute it:** term 2 differing by more than ~1.3x between arms. That would
mean the replay itself is slower in training, which points at pool geometry or block
count rather than at python.

**Why the direction matters more than the number:** if the residual carries it, this is
a submit/poll/python problem, not a kernel problem, and the entire backward-lever
intuition has been aimed at the wrong half of the process. The 12-row lever stack I
priced at 7.09% of the step is all device-side work.

## The bundle caveat, pre-committed

One process, one card, two engines, config the only variable, identical context
schedule. A positive result **localizes to the bundle, not to a mechanism** — it says
the cause is inside the config and not which field. Reported as localized.

The bisection is fixed now, in writing, at three groups rather than eleven fields:

1. **pool geometry** — `num_blocks`, `max_blocks`, `max_total_tokens`, `max_num_batched_tokens`
2. **slot and batch shape** — `num_slots`, `max_batch`
3. **store and sampling** — `NoPrefixStore`, `kv_fp8`, sampling params

The failure mode this guards against: the reproduction comes back positive and the
grouping quietly becomes "well, obviously it is X" — an eleven-way search wearing a
plan's clothes.

## The consumer

Rollout is 73.8% of a GRPO step and the tick sets the rollout, so a closed gap takes a
step from 85.617 s to roughly 46 s. Nothing else consumes this number: no tier, no
default, no pending reject. That is why it outranks the backward lever stack, which is
7.09% of the step even if all 12 rows hit 1.4x at once.

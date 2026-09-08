# One resource, two consumers: three fixes, three wrong axes — 2026-09-08

**Status:** fixed. Four people wrote a fix for one pool-sizing defect in one afternoon and the
first three were each wrong on a different axis. `main` was red twice.

## The defect

`cli.py` sizes the training engine's KV pool. The eval arms (`gsm8k_accuracy`,
`mmlu_accuracy`, the curve) submit into that **same engine**, and this function never sees
their prompts or their length cap. Every version of the sizing expression left out some part
of what the eval arm asks for.

`BLOCK_TOKENS = 16`. GSM8K's rendered prompt is ~150 tokens. `--max-new-tokens` defaults to
32, `--eval-max-new-tokens` to **2048**.

| version | expression | `--group 8`, default eval cap |
|---|---|---|
| before #318 | `ceil(ctx/16) * 8 + 8` | 520 blocks, needs 1099 — **exhausts** |
| #318 | `ceil(ctx/16) * rollout_batch + 8` | same, plus `--group 2` now exhausts too |
| #320 (mine) | `ceil(ctx/16) * max(rollout_batch, 8) + 8` | 520 blocks, needs 1099 — **exhausts** |
| this | per-consumer, then `max()` | 1144 blocks — passes |

## Each fix was wrong on a different axis

**#318 was not the cause.** `tilerl-48`'s negative control: with `--eval-n 8`, the same
exhaustion happens on the commit *before* #318 (521 blocks) and after (1033). The defect is
older; `--group` defaulting to 8, equal to the eval concurrency, is what hid it. Reverting
#318 would have left it in place on the default `--eval-n 100`.

**#320 fixed the row axis while the shortfall was on the length axis.** `ctx` used
`args.max_new_tokens`; the eval arms run at `args.eval_max_new_tokens`. Taking `max()` over
rows and leaving length alone changes nothing about how many blocks 8 rows of 2048 tokens
need.

**And #320's own tests passed for a reason unrelated to the fix.** `max(2, 8)` handed the
narrow group 4x the rows it used, and that surplus absorbed the length shortfall — an
over-allocation on the wrong axis, covering the right one. At `--group 8` there is no
surplus and the gap is exposed.

**The fourth version (rows back to `rollout_batch`, length maxed) over-allocated.** It
prices the widest row count against the longest sequence, so `--group 16` asks for 2280
blocks where the eval needs 1136. On this path that is not slack — the same memory holds the
gradients, the tape and the optimizer state.

## The correct shape: price each consumer, then max

```python
rollout_ctx = max(map(len, prompts)) + args.max_new_tokens + 64
eval_ctx = max(max(map(len, prompts)), 515 if args.eval_mmlu else 0) \
    + args.eval_max_new_tokens + 64
eval_rows_in_flight = min(rollout_batch, _EVAL_CONCURRENCY)
blocks = max(rollout_batch * -(-rollout_ctx // BLOCK_TOKENS),
             eval_rows_in_flight * -(-eval_ctx // BLOCK_TOKENS)) + 8
```

Two independent (rows, length) pairs, maxed. Crossing the axes over-allocates; maxing one
axis alone under-allocates.

`515` is MMLU's longest rendered prompt — the number the 1024 floor was originally chosen
for. The comment above that floor already said the eval arms submit prompts this function
never sees, and it covered the prompt half while the completion half went missing.

## The `min()` is measured, not assumed

`min(rollout_batch, _EVAL_CONCURRENCY)` says the eval arm cannot hold more blocks than the
engine has slots: `generate_ids` submits `_EVAL_CONCURRENCY` requests, but a submit past
`num_slots` queues inside the engine rather than allocating.

Two explanations for why `--group 2` passed while `--group 8` failed were on the table — a
`ctx` floor effect, and this slot gating — and both predict the same outcome on every arm
that had been run. The discriminating case: `--group 4`, 520 blocks, eval cap **1500**.
4 rows in flight need 384 blocks and pass; 8 rows would need 768 and fail. Measured: passes,
so the concurrency is slot-gated.

**My first attempt at that discriminator had no resolution and looked identical to one that
did.** `--group 4` at eval cap 2048 needs 4 × 130 = 520 blocks against a pool of 520 — 
exactly on the limit, so both hypotheses predict exhaustion. It failed, and a failure there
is evidence for neither. Moving the cap to 1500 put margin on both sides.

## Two negative controls, one of them broken

`tilerl-48` first tested the fix by reverting `cli.py` wholesale. The test went red — with
`ImportError: cannot import name '_EVAL_CONCURRENCY'`. **Red for the wrong reason is not a
negative control**; it proves the test can fail, not that it can detect this defect.
Mutating only the sizing line gave `all 521 blocks in use`, which is the mechanism.

The control run here mutates that one line to the #320 shape and the new arm fails on the
assertion it is meant to fail on:

```
AssertionError: (2, 4, 2048, 72, 266)
assert 72 >= 266
```

## Why a green suite shipped a pool that exhausts under the default command

#320 added three arms — `(group 2, 4)`, `(2, 4000)`, `(16, 4)` — chosen to cover both
branches of the `max()` it introduced. **Every one of them passes `--max-new-tokens`
explicitly, and none passes `--eval-max-new-tokens` at all**, so no arm ran the pair
(wide group, default eval cap) that `tilerl train --rl` uses.

The assertions existed, the branches executed, and the parameter combination was not the
product's. That is the third variant of one failure today, after "an assertion exists but its
branch never executes" and "turning a literal into a computed value empties every test that
asserted the literal". **Branch coverage and configuration coverage are different
properties**, and a suite can have the first while shipping a default that dies.

The new arm therefore leaves `--eval-max-new-tokens` at its default and reads the expected
figure from the parser, because writing a literal cap into the test is the exact move that
let #320 miss this.

## Rules

- **Price each consumer of a shared resource on its own (count, size), then take the max.**
  Maxing one axis under-allocates; crossing the axes over-allocates. Both look like "taking
  the max over consumers".
- **A negative control must fail through the mechanism under test.** Reverting a whole file
  produces an `ImportError`; mutate the one line whose correctness is the claim.
- **An experiment sitting exactly on a threshold has no resolution, and its output is
  indistinguishable from one that does.** Check that the two hypotheses predict different
  outcomes with margin before running it.
- **At least one test arm must run the defaults.** A suite whose every arm overrides the
  flags tests configurations no user has. Read the expected value from the parser rather
  than writing the default as a literal.
- **A fix that passes its tests for a reason unrelated to the fix is indistinguishable from
  a correct one.** #320's arms passed because a wrong-axis surplus covered the right axis.
- **Establish a defect's age before attributing it to the change that exposed it.** The
  negative control (same failure before and after #318) is what stops the next person from
  reverting the wrong commit.

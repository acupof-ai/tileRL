# One engine, two consumers, only one of them sized the pool — 2026-09-08

**Status:** fixed. My own change turned a literal into a computed value, and the new value
was computed from one of the engine's two consumers. `main` was red for several hours and
five PRs could not report a green gate behind it.

## What happened

`cli.py` used to size the training engine's KV pool from a literal 8. I replaced the
literal with the rollout width, because `--group 16` was silently running as two waves of
8:

```python
rollout_batch = max(args.group, 1)
num_blocks = -(-ctx // BLOCK_TOKENS) * rollout_batch + 8
```

That change is correct and the coupling defect it fixed was real. It also made the test
suite red, on a test that had passed for months:

```
tests/test_ledger.py::test_train_cli_writes_manifest_and_is_idempotent
RuntimeError: PagedKvPool exhausted: all 137 blocks in use
```

The failing test runs `--group 2`. The arithmetic:

| | blocks |
|---|---:|
| pool at `--group 2`: `ceil(1024/16) × 2 + 8` | 136 |
| the graph's pad block | +1 |
| **what the error reported** | **137** |
| what the eval arm needs: 8 rows × `ceil(1024/16)` | **512** |

**The rollout is not the engine's only consumer.** `evals()` calls `gsm8k_accuracy` and
`mmlu_accuracy` on the same engine, and all three call sites passed a hardcoded
`concurrency=8` that has nothing to do with `--group`. So the pool was sized for 2 rows and
then asked to hold 8.

`--group` defaulted to 8, which is the eval width, so on every path except that one test
the two consumers happened to agree. **The defect was reachable only where `--group < 8`,
and the whole tree has exactly one such place.**

## Why my own guard could not see it

The same change added `_require_group_fits`, which exists precisely to catch "a consumer
wider than the engine":

```python
if engine.usable_slots < group:
    raise ValueError(...)
```

It compares `usable_slots` against `group`. **The eval arm does not go through `group`, and
it exhausts blocks rather than slots.** The guard and the defect are on two different
quantities.

I wrote a guard for "the consumer is wider than the engine" and covered the one consumer I
had in mind. The question that would have found it is not "is the rollout sized correctly"
but **"how many things submit into this engine"** — enumerating the consumers, rather than
checking the one I had already thought of.

## The fix

`_EVAL_CONCURRENCY = 8` as a module constant, referenced by the three call sites, and the
pool sized for the wider consumer:

```python
pool_rows = max(rollout_batch, _EVAL_CONCURRENCY)
num_blocks = -(-ctx // BLOCK_TOKENS) * pool_rows + 8
```

`max_batch` is deliberately left at the rollout width. Widening it would change how wide
the eval batch actually runs, which is a performance change wearing a bugfix's clothes.

`num_slots` was also proposed (8 concurrent eval rows want 8 slots, and `--group 2` gives
2, so the eval queues and warns). **Rejected:** `num_slots` is the input to `usable_slots`
and therefore to `_require_group_fits`. Raising it to 8 makes that guard silently vacuous
for every `group` in 2..8. Trading a slow path that exists only in one test for a silent
correctness guard is a negative exchange. Fixing the eval side to submit at the engine's
actual width is the real repair, and it is a different change.

## The test caught a branch that the obvious test does not reach

Two sessions independently said no new test was needed: the `--group 2` test was already
red, so it *is* the test for this bug. That is true for regression detection and false for
branch coverage.

I widened the existing pool assertion to three arms and it failed immediately:

```
{(2,4): 520, (2,4000): 2072, (8,4): 520}
assert 520 > 520   ← the group arm
```

**`max(8, _EVAL_CONCURRENCY)` and `max(2, _EVAL_CONCURRENCY)` are both 8.** At `--group 8`
the `group` branch of that `max()` does not execute. So after the fix, `--group 2` goes
green — and it goes green entirely through the constant branch:

| mutant | `--group 2` | `--group 16` |
|---|---|---|
| pool sized on `rollout_batch` (the bug) | **red** | red |
| pool sized on `_EVAL_CONCURRENCY`, no `max()` | **green** | **red** — pool 2x too small |

The second mutant is the one the "no new test needed" position cannot kill. The wide arm is
16 for that reason, not 8.

## Rules

- **Ask how many consumers a resource has before sizing it from one of them.** "Is the
  rollout sized right" has a yes; "what submits into this engine" is the question that
  finds the eval arm. A guard covering the consumer its author had in mind reads as
  covering the class.
- **A guard is only a guard on the quantity it compares.** `usable_slots < group` cannot
  see a consumer that exhausts blocks and never passes through `group`.
- **A test that reproduces a bug is not a test of its fix.** It fails for the bug and
  passes for anything that isn't the bug, including a wrong fix. Enumerate the mutants the
  repro admits.
- **A `max()` tested on one side is not tested.** Turning a literal into `max(a, b)` needs
  an arm on each side; without it the new branch ships unexecuted, which is the
  forward-facing half of "turning a literal into a computed value empties every test that
  asserted the literal".
- **A correct change surfacing an unrelated failure is the normal case, not a sign the
  change was wrong.** Making one thing right runs code paths that never ran, and their
  first execution is where older defects appear.

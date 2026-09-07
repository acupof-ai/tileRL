# The tier wrote 321 MiB and never read a byte — 2026-09-07

## Context

Row 50 PR B (#243) moved the SSD fault-in off the tick: `submit` issues a
prefetch, a reader thread does the `torch.load`, and `lookup` **declines** a
prefix whose fetch is still in flight rather than reading it again on the
calling thread. Every gate was green — 85 e2e tests, three mutation-tested
arms, a CPU parity run — and it merged.

The first card measurement after it came back INVALID:

```
faulted  wall 2.070s  ssd_hits 0  entries 1  recovered 1  prefetches 0
"INVALID": "the faulted arm took 0 SSD hits with 1 entries recovered,
            so whatever it measured was not the tier"
```

## Root cause

**A request looks up its prefix exactly once, at admission, and that is the one
moment its own prefetch is guaranteed to still be reading.**

`submit` queues the prefetch and returns. The next tick calls `_build_plan` →
`_admit` → `_match_prefix` → `lookup`. The fetch has had one tick — microseconds
— so `lookup` takes the decline branch and returns a miss. `_admit` proceeds with
`matched=0`, the row prefills the whole prompt, and the bytes land in `_fetches`
with nobody left to take them.

The decline was correct in isolation and is what keeps the 1.7 s read off the
tick. What was missing is that declining is only sound if something asks again.
Nothing did.

## Why every gate passed

The three arms in `test_e2e.py` all drove `PrefixStore` directly and called
`lookup` **after** waiting for the fetch, or asserted the decline itself. That is
the library contract, and the library was correct: on the card, in isolation,
`prefetch_if_worth_it → True`, then `lookup → 128 tokens, ssd_hits=1`. The bug
lives entirely in the *engine's ordering* of two correct calls, and no test
exercised submit-then-admit against a recovered tier.

Two things made it hard to see from the outside:

- **The write side works.** 321 MiB spilled, marker and fingerprint fine, entry
  recovered on restart. A green half looks like a green whole.
- **The number was plausible.** `speedup_faulted_over_cold: 1.071` against a
  1.738–1.821x baseline reads as a mild regression worth investigating, not as
  "this measured nothing". A wrong number in a believable range survives review
  in a way an absurd one does not. The bench's own `ssd_hits > 0` guard caught
  it; reading the ratio would not have.

**`fetch_waits` was incremented at `kv_cache.py:1135` and exposed nowhere.** The
one counter whose value names this bug exactly — declines with zero hits — was
invisible to `/health` and therefore to the bench.

## Fix

One branch in `_build_plan`: hold a row for a tick while its own prefetch is in
flight, bounded by the deadline already computed two lines above.

It sets the row aside and continues rather than `break`ing. The first draft used
`break`, which is head-of-line and would have stalled every other waiting row for
a read only the held one benefits from — caught in review before it ran.

`fetch_in_flight` walks the same block-aligned ladder `prefetch_if_worth_it`
queues on, so the engine asks about exactly the fetches it started.

## Two git commands that reported success and discarded work

Both happened while landing this fix, both in a scratchpad worktree, both caught by
a count rather than by an error.

**`git stash pop` restored 1 of 5 files and exited 0.** It listed
`scripts/bench_ssd_restart.py` as modified and said "The stash entry is kept in case
you need it again" — which is the only signal that anything went wrong. The engine,
kv_cache and test hunks were silently dropped. The stash survived, so recovery was
`git diff stash@{0}^ stash@{0} -- <paths> | git apply --3way`.

**`git checkout --theirs <file>` on a conflicted file deleted a test.** Resolving the
test-file conflict that way took the incoming side whole and discarded the two
parametrized cases the other commit had added. `pytest --collect-only | wc -l`
reading 86 where 88 was expected is what caught it; nothing errored. A second splice
then dropped a third test that had arrived from an already-merged PR (#247) — same
mechanism, caught the same way.

Earlier the same day, `git checkout scripts/bench_ssd_restart.py` discarded unstaged
edits to that file for the same reason: the command's contract is "overwrite from the
index", and the index did not have them.

**Rule.** After any git operation that merges or restores — pop, `checkout --ours` /
`--theirs`, `apply`, a resolved conflict — count what you expect to have and check it:
`pytest --collect-only -q | wc -l`, or `git diff --cached | grep '^-def '`. A commit
whose diff removes a definition it was not meant to touch is the signal, and it is
visible in one command before the commit, not after.

Where a subsystem has a write path and a read path, a gate that exercises one
proves nothing about the other. Assert the read side by a counter that only the
read side can move — here `ssd_hits`, not wall clock, because wall clock is
identical whether the tier answered or the prefill did.

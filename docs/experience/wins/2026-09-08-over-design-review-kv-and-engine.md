# Over-design review: `kv_cache.py` and `engine.py`, two days of commits

> Adversarial review requested by ckl, 2026-09-08. Target: **code that is reachable but
> larger than it needs to be** — not dead code, which was audited separately (net −37 lines
> across 65,768). Scope: `src/tilerl/kv_cache.py` (1483 lines) and `src/tilerl/engine.py`
> (1712 lines), reviewed in `git log --reverse` order over 27 commits.
> **Nothing changed. This is the list.**

## Authorship caveat, first

Every one of the 27 commits is authored `cklxx`, so **git cannot tell me which are mine** —
the same limitation recorded in `branch-ownership-needs-the-board-not-git`. The one I can
verify from this session is `debf6b3` (#296, the slot warning), and I exclude it. If any
other listed commit is mine, this review is not adversarial on that item and should be
re-run by someone else.

## The named suspicion, answered first: fp8 is one path plus a required twin

ckl's starting point via `tilerl-27`: `45acd87` is +624 lines, the largest single diff in the
window; is the fp8 KV path two paths or one plus a CPU twin?

**One path plus a required twin.** The evidence:

- The dispatch is a single function, `model.py:199 _kv_operands`, branching on
  `has_kernel("paged_attention_fp8")`. There is no second engine path, no second writer
  path, no flag pair.
- `kv_layer` (`kv_cache.py:109`) is the twin: it dequantizes the plane so a reader with no
  sub-f32 type can attend. That reader is the CPU cell, whose C backend cannot codegen fp8
  at all — so this is the parity twin AGENTS.md mandates ("every kernel has a CPU twin"),
  not a duplicate implementation.
- `kv_operands` (`:125`) is the raw-plane accessor for cells that have the kernel. Two
  accessors, two dtypes, one caller that picks between them.

The docstring at `:110-117` already prices the twin (0.1 ms to 87.5 ms per tick, proportional
to `num_blocks`) and says it is correctness rather than performance. **So 624 lines is the
size of the mechanism, not of a duplicated one.** Not an over-design finding.

## Findings

### 1. A `ponytail:` marker whose upgrade has landed — `kv_cache.py:483`

```python
# ponytail: sync reload (torch.load), pinned-ring async prefetch when hit
#   latency bites; raw bf16 spill, fp8 tier-quant is 2x capacity if SSD fills
```

The async prefetch **shipped** in `22ece27` (#243): `_fetch_loop` on a reader thread
(`:572`, `:798`), `prefetch()` queueing through `_rq` (`:763`), `take()`/`discard_fetch()`
for the handoff. The marker still describes the pre-#243 world and names the thing that
exists as a future option.

- **Delete/merge:** rewrite the marker, ~2 lines. The remaining true half is the second
  clause (raw bf16 spill; fp8 tier-quant not done).
- **What is lost:** nothing. The stale half actively misleads — it invites someone to
  "add" a prefetch that is already there, and AGENTS.md's rule is that a `ponytail:` names
  a *live* ceiling.

### 2. `max_pending` is a knob for a bound that was measured not to bind — `kv_cache.py:488`

The parameter's own comment (`:499-507`) is nine lines explaining, with numbers, that it is
**not** what protects the host: the queue drains 3x faster than the workload fills it, a full
queue is 18.3 GiB against 386 GiB of allowed dirty pages (4.7%), and what binds first is
`max_bytes` — 37 of 72 offers evicted in that run. That is #186's verdict.

Reachability, enumerated: `max_pending` is **never set to a non-default anywhere** — not in
`cli.py`, not in `engine.py`'s `build_engine`, not in any test, not in any script. Two probe
scripts mention it in prose only (`probe_ssd_arrival_rate.py:3`, `probe_save_fsync.py:7`).
Contrast `max_bytes`, which a test (`test_e2e.py:1488`) and a probe
(`probe_save_ms.py:39`) both drive, and `min_tokens`, which has a CLI flag.

- **Delete/merge:** ~4 lines (the parameter, the assignment, the guard's second clause at
  `:714`). The nine-line comment becomes a one-line note on `max_bytes`.
- **What is lost:** the refusal path it guards. `spill_kv` refuses when
  `len(self._pending) >= self._max_pending`, and `self.refusals` counts that. If the queue
  is ever the wrong shape on a different device, this is the knob that would be reached
  for — and the measurement that retired it was **one card, one workload**. This is a
  judgment call, not an obvious deletion: I would keep it and shorten the comment, because
  a constant that a future card might need is cheaper than re-deriving the bound.

### 3. A module-level marker the same file contradicts 370 lines down — `engine.py:27`

```python
# ponytail: no preemption/swap — admission is capped at ``max_batch``.
```

`engine.py:396-397`, which I wrote in #296 four hours ago, says the opposite:

```python
# A slot is held from submit() to finish, so usable_slots -- not max_batch --
# is the real concurrency ceiling, and _build_plan's max_batch is unreachable.
```

Both are in the tree. Admission is capped at `usable_slots` (`_admit` returns `False` on
`free_slots < 1`); `max_batch` bounds a *plan's* row count, which the warning at `:403`
exists to say is unreachable when the slot pool is smaller. The first half of the marker
(no preemption/swap) is true and is the ceiling that matters.

- **Delete/merge:** ~1 line — drop the "capped at `max_batch`" clause, keep the
  preemption/swap ceiling.
- **What is lost:** nothing, and this is the same defect class as #296 itself: **a
  consequence stated in prose, in a file whose behaviour moved under it.** I found #296's
  version because a peer hit the warning; this one because I was reading for something else.
  Neither had a test, because neither is a consequence any test asserts.

### 4. Not a defect: the sync `torch.load` beside the async fetch is instrumented, not a half-state

`load_kv` (`:848`) and `load_state` (`:899`) still call `torch.load` on the calling thread,
which looks like the old path left in parallel with #243's new one — an AGENTS.md
no-half-states violation.

It is not, and the code says so at `:847`:

```python
# counted: this is the number that can refute "the fetch became async"
self.tick_loads += 1
```

`tick_loads` reaches `stats()` as `ssd_tick_loads` (`:968`) and **two tests assert it is
zero** on the served path (`test_e2e.py:3331`, `:3380`). So the sync call is a
correctness fallback for the case where no prefetch was issued, and it is gated by an
assertion rather than by hope. Recording this so the next reviewer does not spend the same
hour: **the parallel path is real, and it is the guarded remainder, not the old route.**

### 5. Not a defect: the `bytes_per_token` docstring's 1.969x

`kv_cache.py:134-138` documents "1.969x, not 2.000x, which is what the per-token scale grid
costs". Checked against the code: the fp8 branch adds the scale plane's bytes, so the
docstring's number is derived from the same expression the code evaluates. No divergence.

## What I did not review

`engine.py`'s 1712 lines got the `ponytail:` sweep (9 markers; finding 3 is the one that
does not hold, and three others were spot-checked against the code that would have retired
them — `:452` adopted-prefix rebuild, `:623` TTL sweep, `:1443` `_failed` TTL — all name
upgrades nobody has built) and the fp8 call-site trace, not a line-by-line read. The three largest recent
engine diffs — `3401476` queue-a-prompt (+106/−77), `6053f44` keep-the-graphs, `a43a379` the
publish gate — each carry a bench entry or an errors entry and were reviewed against those
rather than fresh. **A second pass on `engine.py` alone is worth someone's time**, and it
should not be mine: I touched `debf6b3` and reviewed `a43a379`/`f33b0b8` earlier tonight, so
I am not adversarial on that file's recent history.

## Summary

| # | item | `file:line` | lines saved | verdict |
|---|---|---|---:|---|
| 1 | stale `ponytail:` half | `kv_cache.py:483` | ~2 | **fix** — the upgrade landed in #243 |
| 2 | `max_pending` knob | `kv_cache.py:488` | ~4 | **keep, shorten the comment** — retired on one card |
| 3 | `ponytail:` contradicted by the same file | `engine.py:27` | ~1 | **fix** — `usable_slots`, not `max_batch` |
| 4 | sync `torch.load` beside the async fetch | `kv_cache.py:848` | — | not a defect, counter-asserted at 0 |
| 5 | `bytes_per_token` docstring | `kv_cache.py:134` | — | not a defect |

**Two items to fix, ~3 lines.** That is the answer to "how big is this": the two files are not
carrying an over-design problem, and the fp8 diff that looked like the biggest risk is the
mechanism's own size. The largest single artifact in my area — `KvTier`, ~460 of
`kv_cache.py`'s 1483 lines — is off by default and under a REJECT, but the reject is narrow
by its own words ("on the serve path **at this session count**", one card, one commit
`a5b63cb`), and it names the block-granular store as the upgrade that would not inherit the
verdict. **Deleting it on the strength of that verdict would be reading a threshold as a
law**, which is the error this repo has recorded twice.

# A prompt's own publishes evict the prefix it shares — 2026-09-07

> Status: **fixed** 2026-09-07 by REPLACE at the decode publish site. A row retires its own
> previous decode entry when the next lands, so one conversation holds
> `prefill_chunks + 1` entries at any moment rather than one per boundary. Both operands
> below are measured on the CPU target: a shared head survives a gap of `budget − 1`
> publishes (10 on the V100), and one conversation emits ~42 publishes at gen 1024 — not the
> 70 first written here. The card numbers (restart-bench warm ratio, `/health` evictions per
> conversation) are **pending-remote**: the V100 is held by another session's serve child.
> That arm must run `bench_ssd_restart.py --gen 256`, not the default `--gen 8`: a
> decode-boundary publish needs a chain end on a 16-multiple, so at gen 8 there are **zero**
> of them and both columns would read 0 by construction rather than by measurement. The
> bench now captures `prefix_evictions`/`prefix_superseded` and prints that warning itself.
> Removed from OPEN.md.

## Context

`/health` on the V100 read **115 published, 11 resident, 105 evictions**, with the
block pool 91% free — so every eviction was state bytes, and the store turned over
its whole contents roughly ten times while serving a handful of requests.

That is not eviction pressure from *many* conversations. One prompt does it alone.
One conversation does it alone, from **two** publish sites. `_finish_prefills`
publishes at each *chunk* end (`engine.py:998` — a guard on `prefill_from %
BLOCK_TOKENS`, not a per-block loop), and `max_num_batched_tokens` defaults to 512,
so a 2729-token prompt publishes **6** times: ⌈2729/512⌉, which matches #241's
measured 6 exactly. The decode path publishes every 16 *generated* tokens
(`engine.py:1308`), so **64** more at gen 1024. Each of the 70 clones a constant
156,893,184 B snapshot, against a budget that holds **11**.

The volume is on the decode axis, not the prompt axis: at gen 8 there are **zero**
decode publishes, which is why a short-generation probe sees none of this. The live
server's 115 published against 3210 tokens_generated fits 3210/16 = 200 decode
boundaries and not its 4 prefill forwards.

Memcpy time is **not** the problem, and this is a bound rather than a measurement:
64 clones × 156.9 MB at the V100's ~900 GB/s is ≤ 11 ms spread over ~1024 decode
steps. The churn is the problem — 70 publishes into 11 slots turns the store over
six times per conversation.

## Root Cause

The entries a prompt publishes are ordered by length, and **LRU keeps the longest
few**. The shared prefix — a system prompt, a common instruction header — is at the
*start*, so it is the first thing evicted by the prompt's own tail.

Measured, budget for 3 entries, one conversation publishing 8 boundaries:

```
resident lengths after 8 publishes:  [96, 112, 128]
a new session sharing the 32-token system prompt:  MISS
```

**Control, same probe with budget for 99:** the new session hits at length 32, with
0 evictions. So the miss is caused by eviction, not by the entry never existing —
the shared-prefix entry is published correctly and then thrown away.

The intermediate entries are **not** useless, which is why this is not simply "stop
publishing them". For the same conversation they never serve: turn 2 is
prompt + reply + followup, strictly longer, so the longest entry answers (measured:
a turn-2 query hits at 96 of 96). But for a *different* session sharing only the
header, the boundary entry is the only thing that can serve — and that is the case
LRU discards first.

## Fix

Not written. The shape the measurement licenses: eviction must not treat a prefix
another session could share the same as one only this conversation can use. Either
score entries by shareability rather than recency, or stop publishing boundaries
inside the divergent tail while keeping the ones at a shared-prefix edge — which
needs a way to know an edge is shared, and nothing currently tracks that.

**Not quantified, and this is the operand the fix turns on:** what fraction of the
70 publishes sit on a prefix another session shares. Arithmetic over assumed
fractions gives 1.4% useful at 0% sharing and 50.0% at 50% — a range so wide it
decides nothing, and the sharing rate has never been measured here. It comes from
traffic, not from a card. A fix priced against an assumed rate would be priced
against nothing.

## The operand is an interval, not a rate — 2026-09-07

The paragraph above asked for the wrong quantity, and `scripts/probe_shared_prefix_lru.py`
says which one it is. The premise it missed is one line: **`kv_cache.py:1120` moves a
matched entry to the MRU end**, so the shared head is only the LRU victim while nothing
is hitting it. Three arms, budget 3, one conversation publishing 8 boundaries:

| arm | head still hits | resident lengths |
|---|---|---|
| no hit on the head | False | `[96, 112, 128]` |
| one hit, right after the head is published | False | `[96, 112, 128]` |
| a hit after every publish | **True** | `[32, 112, 128]` |

The first arm reproduces this entry's own `[96, 112, 128]`, so the probe sees the
eviction that was measured rather than a new one.

**A single hit at any position fails — including one after the last publish.** That is
the mechanism, and it is not "the refresh is too weak": a lookup refreshes only an entry
that is still resident, and once the head is gone the lookup is a miss, which restores
nothing. So the quantity is the *gap between* arrivals, not their recency or their share.

Sweeping the gap against the budget gives an exact relation, asserted rather than
eyeballed:

| budget | max gap that keeps the head |
|---:|---:|
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |
| 6 | 5 |
| 8 | 7 |
| 11 | **10** |

`interval = budget − 1` at all six points, with no fitted term — so the V100's 11-entry
budget tolerates a gap of **10 publishes**. One conversation emits ~42 at gen 1024 (the
boundary-skip section below corrects the 70 written above), so a shared head survives only
if a second session arrives inside every 10 of them, about **4 arrivals per conversation**.

That is a condition a live server can be tested against, which the 1.4%-to-50% range
never was. It also re-shapes the fix: the failure is not that LRU misprices
shareability, it is that **one producer's publish rate outruns any consumer's arrival
rate** — ~42 publishes against a 10-publish window. Scoring by shareability would help
only if it also stopped the tail from filling the budget 4 times per conversation, so
throttling the decode-boundary publishes is the cheaper half and should be priced first.

## Rule

An LRU over entries of *increasing* length evicts the shared prefix first, because
the shared part is the oldest part. Recency and shareability point in opposite
directions when one producer emits a nested family of keys.

Second, from asking for the wrong operand first: **"what fraction shares this prefix"
and "how often does a sharer arrive" are different questions, and only the second one
has an answer the code can be tested against.** A rate looks like the natural operand
because the policy is described in terms of shareability, so the missing number gets
named after the policy rather than after the mechanism. The mechanism here is a
refresh that reaches only resident entries, which makes the binding quantity an
interval — and an interval has a threshold (`budget − 1`) where a rate had only a
range. When a missing operand yields a range too wide to decide anything, suspect it
is the wrong operand rather than an unmeasured one.

## The boundary skip: 44% of the publishes this entry counted never happen — 2026-09-07

Before pricing a fix, the churn was re-read off the live server. It is about **half** what
the paragraphs above state, and the reason is in the guard, not the arithmetic.
`engine.py:1313` tests `i == last and materialized % BLOCK_TOKENS == 0` — only a chain's
**last** accepted token. A chain whose length does not divide `BLOCK_TOKENS` steps over
boundaries silently.

Live V100 serve child (pid 2977356, 6h27m up, `--depth 1` read from its argv), `/health`,
no card taken:

| quantity | value |
|---|---:|
| tokens_generated / decode_forwards | 5410 / 3047 = **1.776 tok/fwd** |
| spec_accepted / spec_drafted | 2365 / 3047 = **0.776** |
| 16-boundaries that exist | 5410 // 16 = **338** |
| `prefix_published` | **188** |
| never published | **44%** |

The dependence is on **divisibility**, not on tok/forward: at a fixed chain length the count
is 338 when the length divides 16 (1, 2, 4, 8) and about 338/length when it does not
(3 → 112, 5 → 67, 7 → 48). At `--depth 1` a tick emits one token plus the drafted one when
accepted, so the chain is Bernoulli — and `1 + 0.776 = 1.776` reproduces the measured
tok/forward exactly. Simulating that distribution gives 163–214 landings, mean 190.6,
against the observed 188: **1.4% off**. `scripts/probe_publish_boundary_skip.py`, with a
negative control — 338 falls **outside** 163–214, so the simulation can tell the skip from
the model it replaces.

**Two earlier models matched 188 to about 1% and were both wrong**, which is the part worth
keeping. `gen/16/tok_per_fwd` = 190.4, a 1.3% match and pure coincidence, since no *fixed*
chain length yields 1.776. Then uniform chains of 1..W bracketed 188 at W=3 — impossible,
because the server runs `--depth 1`; I had asserted W=3 from a mean before reading the argv,
and the assert I wrote to protect the claim is what failed. Agreement at one point is not a
model; the accept rate reproducing tok/forward is what pins this one.

So one conversation emits about **42** publishes at gen 1024 (6 prefill + ~36 landed), not
70 — and the fix has less to throttle than it was credited with.

## The fix: replace, not accumulate

Neither end-of-turn nor a stride, and the reason is the state invariant. A publish must land
**on** a boundary: `insert` refuses a partial block (`kv_cache.py:1045`), the lookup ladder
walks 16-multiples, and a sequence end is aligned only 1 time in 16 — so "publish once at
`_finish`" would either write an unfindable length or slice an entry below its snapshot.
`_finish` also calls `_release`, which frees the blocks, so there would be nothing left to
publish. Verified in the code before implementing, not assumed.

What ships instead: when a row publishes at a decode boundary it **retires its own previous
decode entry**. Only the longest of a row's nested keys can serve it again, so the previous
one is dead the moment the next lands.

| | publishes/conversation | live entries/conversation | turn-2 loss |
|---|---:|---:|---|
| before | ~42 | one per boundary, ~37 | — |
| after | ~42 (unchanged) | **prefill_chunks + 1** | **none** |

Turn 2 still matches the longest boundary crossed, and a mid-generation sharer still finds
the latest entry — which is what a stride policy would have cost. Every stride is strictly
worse on both axes at once: it publishes more entries *and* matches fewer tokens
(at 2729/1024, stride 256 loses 160 tokens of turn-2 match while still sitting at 8.7
publishes).

Two details that are defects if got wrong, both from reading rather than guessing:

- **Retire after the insert.** The two entries share blocks, and `retire` frees pages by
  refcount, so dropping first would free pages the new entry is about to retain.
- **`superseded`, not `evictions`.** `_drop` bumps `evictions`, and routing a replace
  through it would inflate the counter `/health` exists to show pressure with. A separate
  counter, published as `prefix_superseded`; the routing gate covers it (dropping the wire
  line fails `test_kv.py:714`, run as a mutant).

**Only decode entries are retired, and the prefill ones deliberately stay.** A row's
chunk-end publishes are the entries a *different* session's shared header matches — they
are the short prefixes this whole row is about protecting. Retiring those would fix the
count and lose the thing being counted. 6 per conversation is inside the budget, so there
is nothing to reclaim there.

## What `retire` does to a prefix two rows share — and a gate I wrote that could not fail

`retire` matches by **tokens**, and `insert` dedups on them, so two rows publishing an
identical prefix share **one** entry and either row's retire removes it for both. Measured
at the store level: two inserts of the same tokens → 1 entry, one `retire` → 0, and the
other row's `lookup` misses.

That is a **bound, not a bug**, and the reason is worth stating. A second retire of the
same tokens is a no-op returning `False`, so refcounts cannot drift; and the entries a row
retires are its own decode-*tail* boundaries, which is the trade this fix makes on purpose.
The cost lands only on a third session sharing that tail.

The part worth recording is how I nearly shipped a green gate for it. I first wrote an
engine-level arm — two identical prompts at temperature 0, asserting every resident entry
stays matchable — and it passed. Then the mutant that should have broken it (ignore
`_publish_prefix`'s return so a deduped row also retires) **also passed**. The arm could
not fail: with identical tokens, "A's entry" and "B's entry" *are the same entry*, so the
second retire is the no-op above and the mutant is benign. I had built an elaborate
scenario around a hazard the code cannot reach, and the passing test was measuring that
fact rather than any guard.

Replaced with the store-level arm, which has mutants that do fire — routing `retire`
through `_drop`'s `evictions` fails the counter assert, and making a missing entry report
success fails the no-op assert. Both run, each red on its own line.

## Gates


- `test_one_conversation_holds_one_decode_entry_at_every_point_in_time` — counts store
  entries **per tick**, not at the end, because the old behaviour converges to the same
  final contents once LRU has churned; only a point-in-time count separates "published one
  entry 40 times" from "held 40 entries".
- **Non-vacuous first**: the run must publish more than once and `superseded` must be
  non-zero, or a store that publishes nothing satisfies any bound.
- **Mutant, both directions.** Removing the `retire` call fails the non-vacuous assert
  (`superseded == 0`); relaxing that assert so the bound itself is reached, the same mutant
  drives entries monotonically to **6 against a bound of 5** — per-tick counts climb
  1→2→3→4→5→6 while the fix holds flat inside the bound. Run, not reasoned.
- The store-level `#252` xfail stays xfail **by construction**: it calls `store.insert`
  directly, so it never reaches the engine's retire. The fix is at the publish site, and a
  store-level test cannot see it. Said here because "the xfail goes strict-green" was the
  plan, and it is the wrong gate for this fix rather than a missing one.
- `test_kv.py` + `test_e2e.py`: 110 passed, 1 skipped, 1 xfailed.

## Rule

A guard that tests one element of a batch does not fire once per element. `i == last`
turned a per-16-tokens publish into a per-16-tokens-**that-a-chain-lands-on** publish, and
every count derived from the token range was 1.8x high. When speculation, batching, or any
form of multi-item commit sits under a modular condition, the count is set by divisibility
against the commit width, not by the range.

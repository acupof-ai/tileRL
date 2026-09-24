# temp-0 greedy output depends on the traffic before the request — V100 sm70, 2026-09-24

> Status: open — the PR that resets `ticks_since_refresh` at admission removes this line.

Two separate defects, found while diagnosing an unrelated crash in the `#805`
production cutover (V100 switched to `min0` on 2026-09-24). The crash turned out
to be intermittent and is tracked separately.

Measurand for both: the served content for **one identical request** — the same
prompt file, `temperature=0`, `max_tokens=256`, `enable_prompt_thinking=false` —
issued repeatedly against a single running service. The full content is saved per
run (an earlier version of this probe printed only a 60-character head, which is
why the first observation could not be classified after the fact).

---

# 1. A new request inherits the previous requests' sparse refresh phase

> Status: open

## What happens

`SparseRuntime.ticks_since_refresh` is engine-wide. It is initialised to 0 once at
construction (`sparse_runtime.py:120`), advanced on every decode tick, and reset
only when a refresh itself fires (`:307`). Nothing resets it at admission.

A sparse decode tick is eager — it re-scores ALL candidate pages instead of using
the resident hot set — exactly every `SPARSE_REFRESH_TICKS` (8) ticks. Which tick
that is depends on the counter's value when the request starts. So a request's
served tokens depend on the decode ticks the traffic before it consumed.

## Measured

Eight identical 32k requests, greedy, to one `min0` service, `--slots 4`,
preceded by a short chat call and an aborted SSE call. The complete tracer wrapped
both phase-advancing paths (`build_rows` and `run_decode_graph`) and logged
`phase_before`/`phase_after` on every tick. Entry phase read at each request's
first decode step, `T` = that request's decode forwards:

| run | entry phase | T | T mod 8 | class | chars |
|---|---|---|---|---|---|
| 1 | 3 | 139 | 3 | C0 | 879 |
| 2 | 6 | 58 | 2 | C1 | 401 |
| 3 | 0 | 40 | 0 | C2 | 286 |
| 4 | 0 | 40 | 0 | C2 | 286 |
| 5–8 | 0 | 40 | 0 | C2 | 286 |

Three distinct entry phases, three distinct output classes, one-to-one in order.
Runs 3–8 are byte-identical.

The relation is `next entry = (entry + T) mod 8`, and a refresh fires when the
incremented counter reaches 8, so tick *k* is eager iff `(entry + k) mod 8 == 0`.
The fixed point is entry 0 with `T mod 8 == 0`: the phase returns to itself and
the output stops changing.

**Out-of-sample check.** A second run of a structurally identical sequence
produced a *different* structure — a period-4 cycle, `[966, 980, 968, 1215]`
repeating, with entry phases `[0, 5, 3, 2]` cycled and `T = 141, 142, 135, 134`.
There `141+142+135+134 = 552 = 69 x 8`, so the phase returns to its start after
four requests rather than one. The period `8 / gcd(8, T mod 8)` therefore ranges
over 8, 4 and 1 depending on the individual `T mod 8`; both observed structures
fall out of the same rule, and the second was predicted before it was run.

## The slot is not the cause

The period-4 run first looked like slot inheritance: the period equalled
`--slots 4`, and `LinearStatePool.alloc_slot` (`kv_cache.py:568`) zeroes
`states[slot]` and `conv_windows[slot]` and nothing else. A runtime trace of every
`alloc_slot` return refutes it — over the whole lifetime the sequence was
`['4','3','3','3','3','3','3','3','3','3','3']`: the first request takes slot 4
and **every later one takes slot 3**, while the output still cycles with period 4.
A constant slot cannot produce a 4-cycle.

`win_parity` (the conv double-buffer plane index, also not zeroed in `alloc_slot`)
can carry at most two classes across a sequence — it advances by one per decode
tick, so its entry value is determined by the parity of the accumulated `T`, which
here is `[q, q^1, q^1, q]` cycled, putting runs 1 and 4 in the same class when they
are C0 and C3. It is not the discriminator for these observations either. It is
logged in the tracer and needs no separate claim.

## Why it matters

A served configuration has no temp-0 determinism: the same request to the same
service returns different tokens, and what changes it is traffic that has nothing
to do with the request. Any same-service comparison of two configurations is
contaminated unless the phase entering both arms is controlled — measured values
here differ by up to 3x in content length (879 vs 286 characters) from the phase
alone.

## Fix

Reset the counter at admission when there is no other decode row in flight, so a
fresh request always starts at phase 0. `engine.py`, in `_admit`, at the sparse
attach site. At `B>1` the cadence stays batch-global (the cadence is a property of
the batch's tick stream, not of any one row); the code carries a one-line note
marking that and the condition for revisiting it.

## Gate

`tests/test_sparse_refresh_phase.py`. A predecessor request is drained first and
the test asserts its leftover phase is non-zero — without that, the assertion
cannot discriminate, because 0 is also what the unfixed tree reports. Then a
second call is submitted and its phase after the first pure decode tick must be 1
(reset to 0, then one increment).

**Negative control:** with the reset removed the same assertion reads **6** — the
inherited 5 plus one. Confirmed red on the same assertion, not on a different one.

## Reproduction

```
# one service, production launcher, min0, --slots 4
bash ~/stepdiag/phasecycle.sh      # chat -> SSE abort -> 8x 32k, both-path tracer
# per-request entry phase from serve.log; classes from run1..run8
```

---

# 2. `--slots 1` degenerates at 32k

> Status: open

## What happens

Same prompt, same service shape, only `--slots` changed from 4 to 1:

| run | wall | content |
|---|---|---|
| 1 | 329117 ms | `'Based'` followed by 255 × `!` |
| 2 | 17731 ms | 256 × `!` |
| 3 | 17093 ms | 256 × `!` |

All three `finish_reason='length'` at 256 tokens. At `--slots 4` the identical
prompt yields 879–1215 characters of coherent text, so this is degeneracy, not a
different-but-valid continuation. 256 tokens producing 256 characters means one
character per token — a single repeated token.

The runtime trace shows `ALLOC slot=0` **twice** and `ALLOC slot=1` once, where
the `slot=1` allocation comes from a different (2-slot) pool created before the
serving pool. At `--slots 1` the warmup/capture pool and the serving pool appear
to land in the same slot; that is the leading explanation and it is unverified.

## What is measured about the geometry

`build_engine` sizes the pool as `num_slots + pad`, where `pad` is 1 when the
decode graph is on (the replay's padding row owns a slot). Read on the CPU tiny
model, `num_slots` per request against the number of free slots at build:

| `--slots` | graph off | graph on |
|---|---|---|
| 1 | pool 1, free 1 | pool 2, free 1 |
| 2 | pool 2, free 2 | pool 3, free 2 |
| 4 | pool 4, free 4 | pool 5, free 4 |

So at `--slots 1` with the graph on the pool holds 2 slots and exactly **one**
is usable by a request — the padding row is the other one. Which slot index the
request and the padding row land on is not fixed by this table, and the trace
above is the only evidence on that.

What this does **not** establish: that the padding row and the serving request
collide, or that any collision is what produces the repeated token. The CPU tiny
model did not reproduce the degeneracy, so the mechanism is still open and named
as such.

## Why it matters

`--slots 1` is a legitimate-looking configuration that silently produces garbage
rather than failing. It also makes the slot-vs-phase discrimination harder: the
cell that would have held the geometry fixed produces degenerate output, which is
why the discrimination in finding 1 was done with a slot trace at `--slots 4`
instead.

## Not the cause of the slowness

Only run 1 (329 s) is slow, and that is the cold first-32k on a fresh service — an
order of magnitude slower than subsequent runs is the pattern seen at `--slots 4`
too. Runs 2–3 (17–18 s) are *faster* than the `--slots 4` runs (24–33 s). So
degeneracy is the `--slots 1` property; slowness is not.

---

## Notes for whoever picks this up

- Save the full token sequence, not a head. The first observation here was
  unclassifiable because the probe truncated to 60 characters.
- The sparse prefix cache is **not** the explanation for the cycle: the
  `sparse_prefix_hits` counter was non-zero across these runs, and a cache effect
  would not depend on the phase counter.
- A tracer that wraps only `build_rows` misses 7 of every 8 decode ticks — the
  graph runner increments the counter itself (`sparse_runtime.py:839`) and never
  calls it. Wrap `run_decode_graph` too, or the entry phase has to be derived from
  the per-request `T` totals instead of read.
- The same investigation found this config can crash with a CUDA illegal memory
  access at request close (1 of 2 runs that reached the close depth). The two may
  share a cause; that is a hypothesis and is tracked separately.

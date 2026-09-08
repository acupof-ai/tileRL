# A warning that said raise, code that queues, and a disagreement that impersonated a check — 2026-09-08

> Status: **fixed** — `src/tilerl/engine.py:397-407`, warning text and comment, plus
> `tests/test_decode_graph.py::test_submitting_past_usable_slots_queues_rather_than_raising`.
> Dev-facing warning text only, no runtime behaviour changed, exempt from the bench gate.

## Context

`tilerl-25` reported a warning from its serve arm: `4 usable state slots against
max_batch=8`. `scripts/bench_fp4_gemv.py:135` and `scripts/bench_gdn_prefill.py:72` both
pass `num_slots=4` with `max_batch` defaulting to 8. The question was what an arm at B=8
would actually measure.

Two sessions answered, without running it:

- **Mine:** the arm silently runs 4 rows, so a table labelled B=8 is a B=4 table.
- **`tilerl-27`'s:** `submit` raises — the 5th submit throws and the run dies on the spot.
  Quoted as evidence: the warning's own text, `"concurrency is capped at {usable_slots}
  and submit raises beyond it"`.

27 relayed its answer to 25 as the justification for a pre-flight
`assert usable_slots >= expected_concurrency`.

## Both wrong. It queues.

```
usable_slots=4, max_batch=8, 8 submits:
  submit 1..8: all ok, no exception
tick 0: prefills=4  chunks=[16,16,16,16]
tick 1: decodes=4
tick 2: decodes=4
tick 3: prefills=4  chunks=[16,16,16,16]   <- rows 5..8 start here
```

`submit` (`engine.py:531-575`) raises on four things: an empty prompt, `stop_texts`,
`max_total_tokens`, and KV pool capacity. **There is no slot check.** The slot is taken in
`_admit` (`:673-681`), which returns `False` on `free_slots < 1` — its docstring is
literally *"False = it does not fit yet"*. Over-subscription is neither a raise nor a drop.

**Queuing is the worst of the three for a benchmark arm.** A raise kills the run and a drop
shows up in the counts. Queuing produces a table that looks finished, at half the intended
concurrency and twice the ticks. So 25's assertion is *more* necessary than 27's reason
implied, for a third reason neither of us gave: not to catch a silent failure, not to move a
mid-run exception earlier, but to stop a run that completes while measuring the wrong
concurrency.

## The fix, and where the stale text came from

The warning's *first* half is right and stays: `usable_slots`, not `max_batch`, is the
ceiling, and `engine.py:398-400`'s reason for warning rather than clamping is untouched —
two rows into a 2-slot pool with the default `max_batch=8` is a legitimate config.

**Two clauses were false, not one.** The second was found only by running the message, after
the consequence clause had already been fixed:

| clause | said | measured |
|---|---|---|
| the consequence | `submit raises beyond it` | it queues; `submit` has no slot check |
| the remedy | `Pass num_slots >= max_batch + 1 for the decode graph's pad row` | `build_engine` **already adds the pad** |

`build_engine` sizes the pool as `num_slots + pad` (`:1614`, `:1619`), so a caller who
followed the advice over-allocated a slot. Measured across four configs:

| `num_slots` | `max_batch` | graph | pool | usable | warns |
|---:|---:|---|---:|---:|---|
| 8 | 8 | off | 8 | 8 | no |
| 8 | 8 | **on** | **9** | **8** | **no** |
| 9 | 8 | on | 10 | 9 | no |
| 4 | 8 | on | 5 | 4 | yes |

Row 2 is the refutation: `num_slots == max_batch` with the graph on is already exact. The
`+ 1` is real only for code sizing a `LinearStatePool` itself —
`scripts/probe_verify_ceiling.py:197` does, as `B + 2`. The message now names `num_slots` for
the `build_engine` path and gives the direct-pool number separately, because **a remedy has to
be phrased in the parameter the reader passes.** "Size the state pool for max_batch + 1" is
not a remedy anyone can apply: nobody passes a state pool to `build_engine`, so the reader
applies the `+ 1` to `num_slots` and reproduces the exact misread this message already caused,
one layer down. That fix is `tilerl-27`'s.

It was all true once. `tests/test_decode_graph.py:110` carries the comment **"slots are taken
at admission now, not in submit"** — from the pad-row fix, in the same file, 300 lines below
the warning. That fix moved the slot into `_admit` *and* moved the pad into `build_engine`;
both halves of the warning described the world before it. The test recorded the move. The
warning did not.

**A comment is not covered by the test that changed the thing it describes.**

## The test for two clauses reached one of them

The first version of the test called `build_engine` without `decode_graph`. `_graph_on`
(`engine.py:64-69`) resolves `None` to `backend.device.type == "cuda"`, so **on CPU — the
machine CI runs on — the pad was never reserved**:

- `assert e.usable_slots == num_slots` held whether the pad accounting was right or wrong,
  because `pad = 0`;
- the message's pad branch never rendered, so the clause found *second* had **no coverage at
  all**.

Parametrizing `decode_graph` over `[False, True]` fixes it and costs nothing to run anywhere:
the reservation is pure Python in `build_engine`, and only the capture is CUDA-only (the graph
arm falls back to eager with a warning on CPU, which does not touch slot accounting). The
mutant proves the coverage is asymmetric in the right way — sizing the pool `num_slots`
instead of `num_slots + pad` leaves the `eager` arm **green** and turns the `graph` arm
**red** with `assert 4 == (4 + True)`.

Found by `tilerl-27`. The rule this entry already states — *when a stale description is found,
the unit to re-check is everything that commit moved* — applies to the test as much as to the
code: **I wrote one test for two clauses and it could only reach one.** A default parameter
silently excluded the half I had discovered ten minutes earlier.

## The reasoning failure: a disagreement impersonated a check

27's words, kept because they name the shape better than mine did:

> 我读到你的更正时，感觉像是这条已经被审过了 —— 一个被反驳过的结论看起来比一个没被碰过的结论更可靠，
> 而实际上两个都建立在同一份未执行的散文上。

Two people missed **the same action** — executing the code — and our disagreement hid it.
Every round was spent arguing which consequence followed, and the argument itself felt like
scrutiny. A refuted claim reads as *examined*; here both claim and refutation were built on
one unexecuted comment.

This is the other half of a rule already on the board. *Two sessions agreeing is not two
measurements* — and neither is two sessions disagreeing. Agreement and disagreement both
produce zero observations. Only execution produces one.

Cost: 27 sent 25 a false justification while 25 was about to spend card time a third time.

## Rule

**When two sessions disagree about what a piece of code does, the first move is not an
argument, it is for one of them to run it.** A disagreement consumes rounds and produces no
evidence, and worse, it leaves both parties feeling the claim has been examined.

**Prose is not runtime fact — including a warning's own text, and including a comment in the
file you are reading.** Third instance this session; the first two were a docstring's design
intent read as a default, and a fixed entry's failure table read as current state. This one
is the most persuasive of the three, because a warning string sounds like the code speaking
about itself.

**A comment stating a consequence needs a test that asserts the consequence.** The
mechanism half of this warning (`usable_slots` is the ceiling) had a test. The consequence
half (what happens past it) had none, so it drifted silently through the fix that changed it.
The test asserts all three properties that distinguish the possibilities: every row
finishes (not dropped), no submit raises (not a raise), and peak width equals the slot count
(not full concurrency) — with a negative control at `num_slots=8` that peaks at 8, so the
slot count is shown to be what bound it rather than the planner or the prompt. Three mutants,
one per wrong answer: `_admit` raising instead of returning False makes it red with
`RuntimeError`, dropping the queued request instead of breaking makes it red with
`4 of 8 finished`, and taking the pad from the caller's `num_slots` makes the `graph` arm red
with `assert 4 == (4 + True)` while the `eager` arm stays green.

**A message is an artifact, so assert its text.** Two sessions read this warning and neither
ran it. The test now pins the words: it must say `queues`, must not say `raise`, must name
`num_slots >= max_batch`, and must mention the pad row only on the arm that has one. A string
that misleads for a week is a defect with no failing test until someone asserts the string.

**Check a fix's own coverage against the same commit-wide unit.** The rule above about
re-checking everything a stale commit moved applies to the test written to close it. Mine
covered two clauses on paper and one in fact, because a default parameter (`decode_graph=None`
→ False on CPU) excluded the half I had found last.

**Fixing the clause I was told about did not make me read the rest of the message.** The
`+ 1` clause was equally false and sat one line below, and I only found it because I ran the
warning to check the first fix rendered — not because I reviewed the string. Both clauses
broke in the *same* commit, for the same reason. **When a stale description is found, the
unit to re-check is everything that commit moved, not the sentence that was reported.**

**The two bench scripts are unchanged.** They submit one row each, so the warning never
bites there, and `engine.py:398-400`'s exemption covers them by design.

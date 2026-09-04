# The argmax gate was a lottery on the drift, and my first fix for it was worse

**Date:** 2026-09-04
**Arch:** sm90 (found), cpu (refuted, then measured)

## Context

`test_engine_draft_matches_full_context_draft[chunked]` failed on sm90 and on
`origin/main`, so it was pre-existing and not branch-attributable. The assertion is
`int(full.argmax()) == int(got.argmax())` at `test_e2e.py:1038`. The peer proposed
that it is brittle: the two logit vectors are close, and at that position two
near-tied candidates swapped order.

## Numbers

The discriminator is the top-2 margin against the max drift `|full - got|`. Both
arches, same test, same seed:

| case | arch | top-2 | margin | drift | ratio | result |
|---|---|---|---:|---:|---:|---|
| chunked | cpu | 145/146 | 4.6552 | 0.2775 | 16.78x | pass |
| chunked | **sm90** | 226/262 | **0.2863** | **0.3562** | **0.80x** | **FAIL** |
| multirow r2 | cpu | 171/79 | 0.4735 | 0.4172 | 1.13x | pass |
| multirow r2 | **sm90** | 171/79 | **0.2667** | **0.3536** | **0.75x** | **pass** |
| multirow r1 | sm90 | 205/184 | 2.3356 | 1.2206 | 1.91x | pass |
| single | sm90 | 318/48 | 2.4639 | 0.5319 | 4.63x | pass |

Relative drift, sm90: 1.617e-02 to 5.822e-02 against the `rel < 0.1` bound.

**Two positions have drift larger than their margin. One flipped, one did not.**
That is a coin flip in the literal sense — not a badly written test, but an
outcome decided by which side of a 0.28 gap a 0.36 perturbation lands on.

## Two wrong turns, both caught

**Mine, first:** I measured the margins on cpu, found `chunked` had the *largest*
margin and *smallest* drift of any row — 16.78x headroom, the most robust case in
the table — and reported the near-tie reading refuted. It is refuted on cpu. The
margins are per-arch: `tiny()` runs different kernels on the two targets and
sm90's top-2 at that position are different tokens entirely (226/262, not
145/146). I said in advance that cpu might not transfer, which is the only reason
the claim was recoverable.

**The peer's:** the logit deltas they first quoted (~0.01 to 0.21) were pytest's
truncated repr of a 320-wide tensor — the visible head and tail with the middle
elided. The real max drift is 0.3562. A display artefact read as a measurement.
Caught because 0.21 cannot flip a 4.66 margin, so the arithmetic did not close.

**Mine, second, and the one worth recording:** I fixed it by gating the argmax
assert on `margin > drift`. The negative control killed it. Corrupting one logit
by +50 gives margin 2.4555 against drift 50.0787 — **the gate closes precisely
when the draft is most broken**, because a large corruption raises the drift above
any margin. The condition was backwards for exactly the case it needed to catch.

## Fix

Drop the argmax compare. `rel = ||full - got|| / ||full||` already sat three lines
below it and is the assertion that measures the difference rather than betting on
it: sm90's real range is 1.617e-02 to 5.822e-02, and the +50 control reads
**3.235e-01** — caught with 3.2x margin over the bound and 5.6x over the worst
real drift. The argmax token is kept in `rel`'s failure message, where it is
diagnostic rather than load-bearing.

Nothing that was ever real coverage is lost: every genuine break moves the norm,
and every case where argmax carried information also had `rel` well inside bound.

## Rule

A gate on `A > B` where both sides scale with the fault is not a gate. Before
adding a condition that suppresses an assertion, run the control where the
assertion *must* fire and check the condition still opens — a suppressor sized on
passing cases is fitted to exactly the wrong end of the range.

And a per-position margin is a property of the arch's kernels, not of the test.
Measuring it on the cell where the test passes answers a different question than
the one the failing cell asks.

# A draft-bearing sparse prefix hit lands on a different token stream — V100 sm70 / CPU tiny, 2026-09-24

> Status: **PARTIALLY FIXED — the token divergence is NOT closed.** The stored
> boundary hidden defect below was real and is fixed (#827, `5a0c54cc`); device
> measurement after that fix shows a second, separate cause that the fix does
> not touch and that is resident-page history, not the draft. See
> [What the fix did not close](#what-the-fix-did-not-close-2026-09-25) — the
> divergence survives with the **draft off**, so the entry's original
> "draft-off: identical" control does not hold at the served configuration.

## What happens

With a draft, a sparse prefix **hit** produces different temp-0 output from the
same prompt sent as a **miss**, and runs slower. Measured on V100 (impl, at
`260ee7a3` + W1024/R32): miss 36.9 / 40.8 tok/s against a same-process second
submission, which takes the hit path, at 18.7 / 29.5.

The divergence needs **a draft AND a hit together**. Two controls on CPU tiny,
each of which would have been enough to send this down the wrong path if skipped:

| arm | result (CPU tiny, k=2) |
|---|---|
| draft **off**, hit vs miss | **identical** — the trunk logits on the hit path are correct |
| draft **on**, miss vs miss (two engines) | **identical** — the draft path is deterministic |
| draft **on**, hit vs miss | diverges at generated index **12** |

The first row is what ruled out "the hit path corrupts the trunk" — **on CPU
tiny**. Re-measured 2026-09-25 on the served configuration (V100, k=128, W1024,
R32), that first row **does not hold**: with the draft off the hit still diverges
at index 1. The configuration is the variable, so read the row as conditional on
the CPU cell and see the section below. I re-ran the CPU cell at `5a0c54cc` and
both rows reproduce there (`draft=OFF: identical=True`, `draft=ON:
identical=True`), which is why every CPU gate for this defect was green while
the device kept diverging: **the CPU cell cannot fail this test.**

## Root cause

The draft conditions its first tail draft on the trunk hidden at `matched - 1`.

- A **miss** reads that hidden from a full-width prefill forward: `hidden` is
  384 wide, `hidden_prev` is `None`.
- A **hit** restores the entry's snapshot (the state AT `matched`) and
  re-forwards the last page to produce first-token logits. Its newest hidden is
  the **re-forward's recomputation**, which is a different vector: that forward
  starts from the state at `matched`, not from the state at
  `matched - BLOCK_TOKENS` the publisher used.

Both arms are labelled "position 383" and both slice index 383 out of what they
hold, so nothing looks wrong at the call site. The values differ by
**1.305e+03 against a scale of 1310**, i.e. **0.996 relative** — unrelated
vectors, not rounding.

The entry already stores the **correct** hidden: `entry["hidden"]` is
**bit-identical** to the publisher's prefill hidden at that position
(`equal=True`, max|Δ| = 0). The bug is that the hit path never uses it; the value
is read, saved into the row, and then overwritten by the re-forward's output
before the draft runs.

## Why it matters

Any page-aligned prompt re-sent to a draft-bearing sparse service reaches this.
The draft proposes from an unrelated conditioning vector, so proposals degrade,
acceptance falls, and the verify batch shape changes — which is also the speed
loss. The output difference is the visible end of it, not the whole defect.

## Fix

Keep the stored boundary hidden and write it back with the state, after the
re-forward has produced its logits, before the draft step reads it. Measured:
`tokens equal: True` **on the CPU reproduction**. On the served configuration the
restore fires (verified by probe) and the tokens still diverge — see below.

## Gates

Two, because they can fail independently:

- `test_draft_hit_conditions_on_the_stored_boundary_hidden` — the hidden the
  draft will read (built exactly as `DraftHead.step` builds it) must equal the
  full-prefill oracle at the `1e-6` relative level. It also asserts the entry's
  stored hidden really is that oracle, so the gate cannot pass by comparing the
  wrong quantity.
- `test_draft_hit_tokens_equal_a_full_prefill` — the observable consequence, in
  case a future fix restores the right hidden but breaks the token stream
  otherwise.

**Negative control:** with the restore removed the first gate reads
`1.305014e+03 (relative 9.960773e-01)` — red on its own assertion.

**What these gates do and do not establish.** They pin the *value* the draft
reads, which was the defect fixed here, and they do it on the CPU cell. They do
**not** establish that a hit's tokens equal a miss's at the served
configuration, and they cannot: the CPU cell (k=2) does not reproduce that
divergence in either arm (re-measured 2026-09-25, both arms identical). A green
gate here is not evidence about the remaining cause — that cause needs a device
run, and the config it needs is not the one the gates build.

## The xfailed test this fix empties

`test_an_adopted_prefix_redemotes_with_zero_device_bytes` now carries
`pytest.mark.xfail(strict=True)`. It asserts that an adopted prefix page which
leaves the resident union republishes through the zero-byte `share_ref` path.

Measured in that test, on both trees:

| | `dup_refs` (share_ref on adopted pages) | `demoted` | `lifted` | `leftover` |
|---|---|---|---|---|
| `main` | **48** | 0 | 0 | 0 |
| with the fix | **0** | 0 | 0 | 0 |

The second row is why this is "the premise went away", not "the fix broke
share_ref": **no exit path fired at all**, so nothing was substituted. The
follower's tokens still match the prefix-miss oracle exactly.

**Inference, not measurement:** the 48 share_ref calls on `main` most likely
arrive *because* the bug changed the trajectory — the follower generated
different tokens, selected different pages, and that is what dropped the adopted
pages at all. A correct follower behaves like a miss, and a miss never demotes.
That mechanism is plausible and consistent with everything measured, but it was
not isolated; what is measured is the 48 → 0 and the all-zero exits.

A precondition assertion now precedes the `share_ref` claim so the test cannot
pass vacuously, and `strict=True` makes it XPASS the moment the fixture is
rebuilt. The rebuild is tracked in OPEN.md.

## What the fix did not close (2026-09-25)

Re-measured on the served configuration after #827 landed (`5a0c54cc`, V100
sm70, k=128, W1024/R32, q1, depth-1 draft, 37600-token prompt, 512 decode,
temp 0). Same prompt, same tree, one process, miss then hit:

| comparison | result |
|---|---|
| `hitref`: in-process miss vs hit | `first_diff=1`, identical=False |
| `snaprun` vs `baseline`: cross-process snapshot vs full prefill | `first_diff=1`, identical=False |
| `hitref.base` vs `baseline` | **byte-identical** — the full-prefill arm reproduces |
| `hitref.hit` vs `snaprun` | **byte-identical** — the snapshot is faithful |

The last two rows separate the questions: dump/load reproduces the hit exactly,
so the snapshot is not the defect, and the hit really is a different token
stream from a full prefill.

**The fix is active and still insufficient.** A probe wrapping `DraftHead.step`
recorded the hit row's first decode-phase call reading `hidden_from=37599`,
`hidden_prev=None` — exactly the values #827 writes back; without it
`_run_forward` leaves `hidden_from=37584` with a non-None `hidden_prev`. So the
restore fires as designed, and the tokens diverge anyway at the same indices as
before the fix (1 and 13).

**The remaining cause is the resident page set, and it is not the draft.** With
the draft **off** (premise asserted in-process: `e._draft is None`, and
`spec_drafted == 0` after both arms) the hit still diverges at index 1. Measured
at each arm's first decode tick:

| | miss | hit |
|---|---|---|
| resident pages | 359 | **396** |
| candidates per source group | 2336 | **2349** |
| chosen set, 4 groups | — | **all four hashes differ** |

Those resident sets come from different histories: the miss's from its own
prefill's last block selection, the hit's from the 16-token re-forward it ran
after adopting. Under R>1 the sparse approximation depends on which pages are
resident, so the two arms are two legal but **different approximations**, and
the trunk attends different pages from the first decode tick. That is why the
divergence survives turning the draft off, and why it is config-dependent: the
CPU cell (k=2) does not diverge in either arm.

**The shape of the divergence agrees with that reading.** Over 512 generated
tokens, p0 differs in **508** and p1 in **493**, with **no re-convergence** (the
last differing index is 511 in both; the drift grows monotonically rather than
flipping near ties).

The decisive evidence for "different input, not a near-tie flipping" is
**upstream of the first sampling decision, not the 508/512**: the resident sets
already differ at the first decode tick (359 vs 396, candidates 2336 vs 2349,
chosen sets differing in all four groups), which is *before* anything is sampled
from either arm. A near-tie flip is a sampling-time event; this is a different
attention input from the first step. The 508/512 is corroboration of how far the
two streams end up apart, not the proof — the proof is that the divergence is
present before a token is drawn. (Credited to rev-ec, who corrected my weaker
framing of this in review.)

**Recorded as design behavior per coordinator ruling** (94, 2026-09-25), with
the magnitude on the record rather than argued away.

**Optional fix, not attempted:** force the hit arm's first decode tick to be a
refresh tick (promote the full candidate set before selecting), which should
converge the resident sets. Unverified, and it buys correctness at a
speed cost on the path whose whole point is speed. Note for whoever tries it:
the first tick of **both** arms is a captured tick, not a refresh tick
(`do_refresh = ticks_since_refresh >= SPARSE_REFRESH_TICKS`; both arms enter at
0), so the asymmetry is not "hit skips a refresh the miss took" — it is the page
sets the two arms carry into that tick.

**Two measurement traps this round, both mine.** The first probe compared each
arm's *first decode-phase* `draft.step` call and reported `hidden_hash DIFF`;
that comparison is invalid — the arms' tensors cover different position windows
by construction, so the whole-tensor hash differs trivially. Hashing the slice
the draft actually reads is the only valid form, and even then the shared
positions are all *generated* positions (37601+), which differ because the arms
already diverged — an effect, not a cause. Second: I labelled the first tick
`was_refresh=True` from `ticks_since_refresh_before == 0`, which is not the
predicate; both arms read False under the real one.

## Two things this entry does NOT claim

- **Acceptance** could not be measured on CPU tiny: it is 0 in both arms, so the
  counter cannot discriminate there. The V100 numbers, including acceptance, are
  the device-side confirmation and are `pending-remote`.
- The stale draft-block question below is a **separate, unfixed** defect found by
  the same investigation; it does not explain this divergence (zeroing the blocks
  at allocation does not change these tokens) and it is not fixed here.

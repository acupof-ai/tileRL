# A draft-bearing sparse prefix hit lands on a different token stream — V100 sm70 / CPU tiny, 2026-09-24

> Status: FIXED by this PR — the stored boundary hidden is restored with the state.

## What happens

With a draft, a sparse prefix **hit** produces different temp-0 output from the
same prompt sent as a **miss**, and runs slower. Measured on V100 (impl, at
`260ee7a3` + W1024/R32): miss 36.9 / 40.8 tok/s against a same-process second
submission, which takes the hit path, at 18.7 / 29.5.

The divergence needs **a draft AND a hit together**. Two controls on CPU tiny,
each of which would have been enough to send this down the wrong path if skipped:

| arm | result |
|---|---|
| draft **off**, hit vs miss | **identical** — the trunk logits on the hit path are correct |
| draft **on**, miss vs miss (two engines) | **identical** — the draft path is deterministic |
| draft **on**, hit vs miss | diverges at generated index **12** |

The first row is what rules out "the hit path corrupts the trunk": with no draft
there is no difference to explain.

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
`tokens equal: True` on the reproduction.

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

## Two things this entry does NOT claim

- **Acceptance** could not be measured on CPU tiny: it is 0 in both arms, so the
  counter cannot discriminate there. The V100 numbers, including acceptance, are
  the device-side confirmation and are `pending-remote`.
- The stale draft-block question below is a **separate, unfixed** defect found by
  the same investigation; it does not explain this divergence (zeroing the blocks
  at allocation does not change these tokens) and it is not fixed here.

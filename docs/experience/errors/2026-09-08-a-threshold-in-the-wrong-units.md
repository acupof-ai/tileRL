# A threshold in the wrong units — P1's exit gate, 2026-09-08

**Status:** fixed — two thresholds plus `ce_falls`. `reward_rises` needs a ruling, not a fix; see below.

## Context

`docs/roadmap.md:57-58` states P1's exit criteria: GSM8K held-out (500 q) after − before
**≥ +5 pt (SE ≈ 2 pt)**; MMLU (1000 q) after ≥ before **− 2 pt**; tied-group fraction < 50%.
`_finish` (`src/tilerl/cli.py:812-860`) encodes them, and `gates_pass` turns the result into
the manifest's recorded verdict and the CLI's exit code. P1 is the phase the roadmap's
later phases rest on, so this gate decides whether that claim was earned.

Found by tilerl-27's read of the roadmap against the code.

## Root cause

Two metrics thirteen lines apart in the writer are in **different units**, and one gate block
compared both with the same shape:

```
cli.py:575   manifest["metrics"][f"mmlu_{tag}"]  = c / n     # FRACTION
cli.py:588   manifest["metrics"][f"gsm8k_{tag}"] = c         # COUNT
cli.py:590   manifest["metrics"][f"gsm8k_{tag}_total"] = n   # its denominator
```

`tests/test_ledger.py:71,102-103` pin the count with `isinstance(..., int)`.

So `gsm8k_improves` was `after > before` on **counts**: passing needed **+1 correct answer out
of 500 = +0.2 pt**, against the roadmap's own SE of ≈2 pt. A zero-threshold gate on a
symmetric null passes about half the time, so the recorded verdict for P1 could be produced by
noise. `mmlu_holds` used `mmlu_before - 0.03` — the right shape on a genuine fraction, with the
wrong number, a real one-point loosening.

**The units mismatch is the root cause, not "looser than spec", because it made the obvious fix
invisible.** The natural correction — `after - before >= 0.05` — is a **no-op**: on integers
`>= 0.05` is `>= 1` is `> 0`, identical to the gate it replaces. That diff would have changed
the source, passed CI, produced a CHANGELOG line and this entry, and altered nothing. It was
caught by asking what unit the operand was in before writing the diff, not by testing the diff.

## Fix

`cli.py`: `mmlu_floor` to `- 0.02`; `gsm8k_improves` to `after >= before + 0.05 * total`, with
`total` read from `gsm8k_after_total` (falling back to `gsm8k_before_total`) so the bar tracks
`--eval-n` instead of hardcoding 25. The units of each key are named in a comment at the site,
with the roadmap's SE sentence quoted so the +5 reads as a noise floor rather than a taste.

`gsm8k_{tag}` is deliberately **not** normalised to a fraction. Every manifest already written
holds a count under that key, and changing the units retroactively would make old records
silently mis-read by any new consumer; `test_ledger.py`'s `isinstance(..., int)` stays as the
pin that keeps them from drifting. (tilerl-27's call, and it is the lower-entropy one.)

`tests/test_ledger.py::test_p1_exit_thresholds_match_the_roadmap` scores gates through
`_finish` itself. Both arms of each case are asserted, because a pass-only test would have gone
green on the pre-fix code: +1 of 500 must fail, +24 must fail, +25 must pass, a 200-question
manifest must move the bar to +10, MMLU at −2.6 pt must fail and −1.9 pt must pass.

## Verification

**The first mutation run was invalid and its results were void.** I reported both thresholds
as independently killing the test; that came from stale bytecode. The mutation is an
equal-length substitution (`0.02` → `0.03`), so the source **size does not change** and the
mtime lands in the **same second** — and CPython's `.pyc` invalidation compares exactly those
two fields, so it reused the old bytecode. Caught by printing
`_finish.__code__.co_consts`, which held `0.03` while the source on disk read `0.02`: the
running code was not the file. A mutation result is uninterpretable until the *loaded* code is
known to be the mutant.

Redone with `__pycache__` cleared between every mutant:

| mutant | test |
|---|---|
| `mmlu_floor` back to `- 0.03` | **FAILS** |
| `gsm8k_improves` back to bare `after > before` | **FAILS** |
| `0.05` → `0.005` (bar 10x too low) | **FAILS** |
| `>=` → `>` (off-by-one at the bar) | **FAILS** |
| drop the `or gsm8k_before_total` fallback | **SURVIVED** — a real hole, fixed |
| control, unmutated | passes |

The survivor was a genuine gap: no case reached the `_total` fallback. A guard stop can leave
the after-arm's total unwritten while the before-arm's is on the manifest, and a bar computed
from a missing total is `None` — a vacuous pass on the gate that matters most. A case for that
was added; all five mutants now fail and the control passes.

## What the roadmap's "SE ≈ 2 pt" actually refers to

Computed here because the next person to argue about +5 pt will need it. On 500 questions the
**unpaired** binomial SE of the difference is 3.04 pt at p=0.36 and 3.16 pt at p=0.50. The
**paired** SE — `sqrt(b+c)/n` over discordant counts, which is what this eval supports, since both
arms score one `eval_rows` list (`cli.py:479`) — is 1.41 pt at a 10% discordant rate and 1.00 pt at
5%.

The roadmap's ≈2 pt sits between the two: it cites an unpaired magnitude but takes a value below
it. So **+5 pt is defined against an SE that is neither the paired nor the strict unpaired one**.
That is not a defect — +5 pt is conservative under either reading — but once the paired test lands
the bar is roughly 3.5–5 paired SE, stricter than the figure in the roadmap implies. Anyone
proposing to relax it should start from that.

## Also fixed: `ce_falls` recorded a pass it could not withhold

`ce_falls` is `ce_last < ce_first`. The RL path never writes `ce_first`: `cli.py:513-515`
initialises the metrics dict without that key and the GRPO branch (`:679-690`) sets only
`ce_last`; the SFT loop writes both (`:281`). A missing threshold is a vacuous pass by design
(`cli.py:848`), so on every RL run this gate reported `passed` over nothing — **not loose,
unconditionally true**.

Fixed by marking it `skipped` on that path, through the mechanism the block already has, so the
manifest records "not measured" — which is the fact. Two alternatives were rejected as larger:
writing `ce_first` on the RL path manufactures an input to feed a gate, and deleting the gate
loses the half that genuinely works under SFT. (tilerl-27's ruling; it is smaller than either
option I had listed.)

The gate stays live where both values exist, and the test asserts the rising arm still **fails** —
a skip that also swallowed a real regression would be this same defect facing the other way.
Both mutants die: removing the `unmeasured` set, and dropping its term from the `skipped`
expression.

## Still open: `reward_rises` is a diagnostic recorded as a verdict

`reward_rises` is a bare `reward_last > reward_first` — zero threshold, so a symmetric null
passes it about half the time. Both sides are windowed means over `len//4` (`cli.py:680-689`),
which is the mitigation the comment at `:674-675` describes, but a windowed mean of noise is
still noise around zero.

Not fixed here, and the reason is not effort: **the roadmap's P1 exit criteria contain no reward
term at all.** So the question is not what threshold to set but whether this belongs in a verdict
rather than beside it, and that is a ruling. `ce_last` was also a single step (`hist[-1][1]`)
against windowed counterparts — the exact shape `:674-675` warns against, eight lines above the
line that did it — which the skip makes moot on the RL path.

## Rule

**A threshold is meaningless without the units of the quantity it thresholds.** Before changing
a comparison, read the writer of both operands, not the comparison. Adjacent metrics in
different units make the wrong reading the natural one, and the resulting fix is a no-op that
looks like a fix — so the mutation must revert *each* changed threshold separately, or one
live change masks another that did nothing.

**And clear `__pycache__` between mutants.** An equal-length edit within the same second is
invisible to `.pyc` invalidation, which compares only source size and mtime, so the interpreter
runs the *previous* bytecode and the mutation reports the last run's verdict — red or green,
either way about code that is no longer on disk. This is the more dangerous of the two findings
here, because it silently disables the instrument that exists to prove the test works. Verify
the mutant is loaded (`f.__code__.co_consts`) or clear the cache; do not infer it from the
edit.

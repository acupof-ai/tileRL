# My combine divides by zero on a row that cannot happen, and nothing said why, 2026-09-03

> Status: **pinned, no runtime change.** Chasing whether upstream's "W=8 speculation answers
> 0 of 500 GSM8K questions" applies to this branch. It does not — that run's NaN came from a
> geometry a guard now covers, and the guard is in **upstream's** combine, not mine. But the
> same latent row exists here with a **different failure mode**: my `m` init is finite, so an
> all-empty row divides by **exactly 0** and yields inf, where upstream's `-inf` init gave NaN.
> The row is unreachable. That reasoning was written down nowhere, which is precisely the state
> upstream fixed.

## Why I went looking

`origin/main` has moved 70 commits ahead and carries
`e947ce2 W=8 speculation answers 0 of 500 GSM8K questions on the 27B`. Every speed number on
this branch is acceptance-driven — 45.9 tok/s at ctx=1024 is `tok/forward` 2.86 — so a
correctness verdict against wide speculation could reprice the whole line. Read before
rebasing.

**It does not apply, for a reason worth recording.** The follow-up `dbf755c` says the 0/500
run was measured one commit **below** the split-KV NaN guard and re-runs clean on main: 0 NaN
of 393 wide row-ticks, 39 of them at the `n % 64 in [1,7]` geometry the guard targets. So the
catastrophic reading was that branch's own negative control. What survives upstream is
narrower: at W=8 an accepted token is not bit-identical to the unspeculated one, cause not yet
separated from near-tie float nondeterminism.

## The row that exists here

`paged_attention_split_combine` (`kernels.py:769`) ends with `Out = o[d] / l[0]`, where
`l = sum_s w_s · PL_s` and `w_s = exp(PM_s − max PM)`. If every split for a row had an empty
window, `PL` would be 0 in all of them.

**My failure mode differs from upstream's** because the initialisation does:

| | `m` init | all-empty row gives |
|---|---|---|
| upstream | `-inf` | `exp(-inf − -inf)` → **NaN** |
| this tree | `-1.0e30` (finite) | `exp(0) = 1`, `l = 1·0 = 0`, **`o[d]/0` → inf** |

An inf propagates into the softmax of the next layer and comes out as a garbage argmax — the
same class of silent wrongness, reached by different arithmetic. Neither is guarded.

## Why it is unreachable

From the split kernel's own index math (`kernels.py:672-676`):

```
n   = SeqLens[bb] - SeqQLens[bb] + tt + 1
per = ceildiv(n, KVSPLIT)
p0  = sp * per              # split 0: p0 = 0
p1  = min(n, p0 + per)      # split 0: p1 = min(n, per) >= 1
```

`n >= 1` ⇒ `per >= 1` ⇒ split 0 has `p1 > p0` ⇒ it runs at least one tile ⇒ that tile holds
key 0, which every query may attend (`hist >= 0`) ⇒ `l[0] > 0`. Checked for every
`n in [1, 4200)` at both shipped split counts, and for padded query rows, which run the kernel
too (`kernels.py:691`).

**The premise `n >= 1` needed checking, not assuming**, and one form of it is not guaranteed by
the kernel:

- The tightest producer is the draft chain step (`engine.py:987`): `seq_len = pos + 1 + j` with
  `j >= 1` against `seq_q_lens = 1`, so `n >= 2` — margin to spare.
- But `backend.py:690` fills an **absent** `seq_q_lens` with the padded width `s`. Then
  `n = seq_len − s + tt + 1`, and a caller passing `seq_len < s` makes `n = 0` at `tt = 0`.
  Nothing in the kernel rejects it. That constraint lives in the dispatch — `seq_len` is the
  row's post-forward length and cannot be shorter than the queries the forward consumes — so
  the test asserts it rather than trusting it.

## What could not be checked, and why

I first added the direct assertion — recompute `l` from the kernel's own `PM`/`PL` inside
`check_split_attn_parity.py` and require `min l > 0`. It cannot run here:

```
tl.reduce requires a target-specific implementation, but no reduce implementation
is registered for {"kind":"c","tag":"","keys":["cpu"]}
```

The split kernel has **no CPU twin** — a recorded gap, not something this change introduced.
So the end-to-end form of the check is GPU-only, and the GPU is held by the demo server. The
assertion stays in the parity script for the next card run; the gate that runs everywhere is
the arithmetic one.

`# ponytail: arithmetic proof, the min-l assertion runs on the next card run`

## Rule

**A conclusion from another branch has to be re-derived before it is imported or dismissed.**
"W=8 answers 0/500" would have been a reason to stop this whole line; it turned out to be a
negative control for a guard that exists upstream. The dismissal was only safe because the
follow-up entry said which tree each arm ran on — **which is why "measure the tree you mean to
measure" is worth the words it costs.**

Second: **the same latent bug can have a different signature in two trees, so matching on the
symptom misses it.** I went looking for NaN, found none, and would have stopped there. The
defect here produces inf, from a finite sentinel that looks more careful than `-inf`.

## Gate

203 passed, 4 skipped, ruff clean. Kernel parity re-run after the comment edit (36 passed,
2 skipped). Negative control in the new test: the same arithmetic asked about split **1**
returns 0 tiles at `n=1`, so the property is specific to split 0 and the assertion is
load-bearing. No GPU used — the direct `min l > 0` assertion is staged in the parity script
and cannot run on this machine.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | — | — | — | upstream's 0/500 GSM8K, applies here? | **no** — its own negative control |
| 2026-09-03 | (this) | — | — | — | my `m` init vs upstream's | **-1.0e30 (finite)** vs `-inf` |
| 2026-09-03 | (this) | — | — | — | all-empty row here | **`o[d]/0` → inf** (not NaN) |
| 2026-09-03 | (this) | Mac | — | — | split 0 tiles, every `n` in [1, 4200), ks 32 and 16 | **>= 1** |
| 2026-09-03 | (this) | Mac | — | — | tightest producer `n` (draft chain, engine.py:987) | **>= 2** |
| 2026-09-03 | (this) | Mac | — | — | unguarded premise | `seq_len >= s` when `seq_q_lens` is absent |
| 2026-09-03 | (this) | Mac | cpu | — | direct `min l` assertion | **cannot run** (no CPU twin for `tl.reduce`) |

# A README number with no measurement record, and the gate that can only falsify

**Date:** 2026-09-05 · **Class:** error · **Where:** `README.md:22`, `README.md:89`,
`tests/test_docs_links.py`

## Context

`README:89` states: *"Every number above sits in a dated entry under
`docs/experience/`."* That is an absolute claim over an enumerable range, so it can be
checked rather than trusted.

## Root cause

`43.2 tok/s wall over the network` (README:22) appears in **no** dated entry. Its only
source is the body of `cb1b59e` (#74), the commit that added the line — and that commit
touched `README.md` alone, so it satisfied the every-runtime-change-gets-a-bench-entry
rule *by not being a runtime change*. The body says "median of three runs measured from
this machine over the network, not on the pod"; those three runs left nothing behind.
`git grep "over the network" origin/main -- docs/` returns zero.

Searching all history for a `43.2` used as a rate finds exactly one, in a table
`fa8634a` **deleted**: the DFlash depth-2 verify column. Unrelated quantity, same
digits — which is the same coincidence the gate's own limitation is about, below.

Measured across the claim's whole range:

| | |
|---|---|
| distinct decimals in README | 23 |
| dated entries searched | 285 |
| absent from all of them | **1** (`43.2`) |
| false positives | **0** |

#87 replaced the line with `50.0 tok/s decode-only, 46.3 wall measured from the pod with
RTT outside the window; 19 GB of weights off disk`. Both rates come from
`wins/2026-09-04-the-pages-rate-is-50-and-my-39-was-a-rejected-instrument.md` (four runs
plus median; 46.3 reconciled against a recorded 46.1 at 1.004x), and the 19 GB from
`errors/2026-09-02-serve-never-sized-its-kv-pool.md:90`.

**What the new line deliberately omits is the more useful half.** The weights' resident
VRAM is not stated, because nobody has measured it on its own: it lies between the 19 GB
on disk and the 23.2 GiB of whole-process residency (`nvidia-smi
--query-compute-apps`, pid 2016848, held all night on this V100 — that figure includes
the KV pool and the draft head). Three real numbers, all near 22, any of which reads
plausibly beside a tok/s figure. That is exactly the shape 43.2 had: its *direction* was
right — a path that includes the network is slower than 46.3 — and it had simply never
been measured. Being right about the sign is not evidence.

## The gate, and the direction it cannot run

`test_no_readme_number_is_absent_from_every_dated_entry`. Two-arm negative control:
restoring `43.2` **FAILED**, repeating an already-sourced number stayed green.

**It falsifies; it cannot verify.** A positive match is a string comparison, while a real
confirmation would need semantic association. `README:57`'s "1.6 points apart" (MMLU
significance) matches "~1.6% of f32 peak" in
`errors/2026-08-29-chunkwise-wy-wrongly-ruled-out.md:74` — two unrelated quantities
sharing a digit string. So *the number appears somewhere* says nothing about provenance,
while *the number appears in no dated entry* is a fact about the README.

Integers are excluded: `8`, `27`, `4096` are configuration rather than measurements, and
scanning them returned mostly version numbers and tensor shapes.

## Three rungs, not two

Yesterday's boundary note had two cases. There are three:

1. **Enumerable range, decidable test** — #83 (handler set over exception types), #85
   (doc invocations over the CLI's flags), #86 (one wrapper's answers against another's).
   Two-directional: membership is decidable, so green means the set is complete.
2. **Enumerable range, approximate test** — this gate. The range is 23 numbers against
   285 entries, but the matching relation is a digit-string comparison standing in for
   "is sourced by". One direction only.
3. **No enumerable range** — which attributes an object must have, since a read can be a
   string literal (`getattr(engine, "limits", 0)`). No gate to write; stays a reading
   problem.

The rung decides what a green gate is worth. On rung 2, green means "no number proven
absent", never "every number is sourced" — and a docstring implying otherwise is itself
the defect, which is why the limitation is written into this one rather than left out.

## Rule

A number that lives only in a commit message has no measurement record, and a commit
that touches only docs slips past every entry requirement. When a document asserts that
all of its numbers are sourced, check it — the claim's range is small and the check is
cheap. When the check can only run one way, say which way in the place someone will read
before trusting it.

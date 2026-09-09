# The 09657c0-vs-HEAD spec throughput gap is not a regression — 27B, 2026-09-09

**Status:** closed — accept-or-reject verdict, recorded in CHANGELOG.

## Context

The README's speculation row read **135.5 tok/s** until 2026-09-09, when a re-measurement on
the current sha found **126.5** — a 6.6% gap with no algorithm change. The pod tree the
135.5-era measurements ran from was then found byte-identical to HEAD: the "09657c0" label
named a tarball, not a revision, so every 09657c0 number from that tree was void. This
re-run is the controlled comparison the question needed: the true 09657c0 (96aed09e)
against HEAD (40aa691a), same card, same instrument, alternating.

## What the data says

Interleaved ABABAB on card 3, n=3 paired, every criterion arm at **0 compiles** (warm
TileLang cache):

| arm | role | pair | tok/s | compiles |
|---|---|---|---:|---:|
| accspf18 | A (HEAD 40aa691a) | 2 | 93.36 | 0 |
| accspf19 | B (09657c0 96aed09e) | 2 | 90.66 | 0 |
| accspf20 | A | 3 | 91.78 | 0 |
| accspf21 | B | 3 | 91.55 | 0 |
| accspf22 | A | 4 | 94.45 | 0 |
| accspf23 | B | 4 | 92.25 | 0 |

HEAD mean **93.20**, 09657c0 mean **91.49**. Paired differences d = A−B:
**+2.70 / +0.23 / +2.20**, mean **+1.71 tok/s**.

**The criterion, in absolute units (the form that can fail):**

```
spread of d (max−min) = 2.47 tok/s        2×spread = 4.94 tok/s
|mean d| = 1.71  <  4.94                  → no decidable difference
```

The same criterion was first written relatively, `s = spread/|mean d| = 144.5%`,
`2s = 289%`, and `1.8% < 289%`. **That form cannot fail** — the denominator is the mean
difference itself, so it explodes as the effect approaches zero and the inequality holds
almost regardless of data. It is kept here only as the lesson: a threshold that cannot
give both answers is not a threshold.

**The load-bearing argument is the second one.** A 6.6% regression on 93.20 tok/s is
6.15 tok/s with HEAD *slower*. The observed mean difference is +1.71 with HEAD *faster* —
7.86 tok/s away from the regression prediction, **3.2× the paired-difference spread**.
This is not "no difference detected"; the data is **incompatible with the 6.6% regression
that opened the question**.

126.5 remains the current README reading; **135.5 has no reproducible provenance and is not
restored.**

## Excluded arms

A1 (accspf16) and B1 (accspf17) ran first and are excluded as a pair, pre-registered:

- **B1 compiled 756 kernels inside the timing window.** The baseline tree's kernels differ
  from HEAD's, so the shared TileLang cache was cold for them, and the harness has no
  warmup — compile cost lands in `wall`. The window is `main()`'s bracket around
  `generate()` in `acc_spec_arm_profile.py`:
  `torch.cuda.synchronize(); t0 = time.perf_counter(); generate(engine, tok, prompts, sp, 1); torch.cuda.synchronize(); wall = time.perf_counter() - t0`,
  with nothing warm between engine build and the bracket. A cold baseline reads slow, which
  inflates HEAD's advantage: the dangerous direction. B1's 4.43 tok/s is a compile-tax
  number, not a measurement.
- **Citation collision:** the instrument is `scripts/acc_spec_arm_profile.py`, not main's
  `scripts/acc_spec_prefill_profile.py` — both sides independently created that filename
  (main's is #370's five-bucket B=1 prefill profiler; its lines near 189 are an unrelated
  store-timing wrapper). The pod ran this file under the old name; the rename is
  name-only, content identical except two self-references.
- **A1 was clean: 93.01 tok/s, 0 compiles, within 0.2% of the A2–A4 mean of 93.20.** It is
  excluded *only* because its paired partner was. Reporting this blocks the "you picked
  the favorable samples" reading: A1 independently reproduces the HEAD arm.

B1's compile was not wasted: it warmed the cache that made B2/B3/B4's 0-compile runs
possible. The compile rate *increased* over the run (10.6→18.9/min) rather than decaying —
not a sign of a non-converging cache, but of back-loaded demand: decode ticks at growing KV
length trigger more new variants than prefill. A constant compile rate is only evidence of
non-convergence under uniform variant demand, which a decode workload does not have.

## Provenance

- Both trees stamped: HEAD `40aa691a`, baseline `96aed09e` (`.synced_commit`, read by
  benchrec). The baseline tree is **09657c0 plus `scripts/card_owner.py` and
  `scripts/benchrec.py` copied from HEAD** — pod/dev tooling absent at that sha
  (card_owner.py landed 2026-09-09), without which the wrapper cannot claim a card and the
  instrument cannot stamp its sha. Verified not on the measurement path:
  `grep -rl 'card_owner|benchrec' src/ packages/` → 0 hits in both trees. The deviation is
  printed in every arm's header (`tree_delta`) and in each report's provenance.
- Every arm: card 3, `gpu_at_start` snapshot in the header, 0 compiles on criterion arms,
  spec acceptance 0.569 (A) / 0.5655 (B), deterministic under seed=0 (identical
  accept/draft counts within each arm).
- Raw reports on the pod: `/work/accspf16/` … `/work/accspf23/prefill_profile.json`.

## The absolute numbers answer sha-vs-sha only

**93.20 is not comparable to the README's 126.5.** This instrument was built for the ratio
(its docstring says so): its wall includes host launch overhead, and its spec acceptance is
0.569 against the README arm's 0.774. The perturbation cancels in the paired difference,
which is why the verdict stands; the absolute level is a different population. A number
without its population is tonight's recurring defect — do not put these two side by side.

## Rule

A threshold written as `spread / |effect|` degenerates to "always passes" as the effect
approaches zero; write the comparison in the units of the measurement (`|effect|` vs
`2×spread`) so it can fire in both directions. And: a cold-cache arm in a timing comparison
is not a slow arm — compile cost in the measurement window biases toward the finding the
study is looking for, so pre-register the 0-compile gate before seeing the numbers.

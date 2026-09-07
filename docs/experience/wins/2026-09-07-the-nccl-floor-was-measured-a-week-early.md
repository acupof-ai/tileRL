# The NCCL floor ten sites consume is 20.6 µs, and it was measured a week before the entry saying it wasn't — H20 cards 0+1, 2026-09-07

> Status: closes [the NCCL floor has no instrument](../errors/2026-09-06-the-nccl-floor-has-no-instrument.md).

## Context

`21.5 µs per all-reduce, flat 20 KB → 1.3 MB` is cited in ten places across
`docs/design-parallel.md`, `docs/roadmap.md` and `CHANGELOG.md`. Three whole-step cost
tables, the ring-vs-all-gather verdict at `design-parallel.md:191` and a block-size
crossover take it as an input. The open entry traced every citation to a CHANGELOG line
that *consumes* the figure to derive 2.8 ms, found `scripts/nccl_probe.py` cited by
nothing, and concluded no run existed.

No run existed **in the tree**. Four existed on the pod, dated 2026-08-29 — a week
before that entry was written — and were never pulled back.

## What Worked

Running the probe that was already there, unchanged, at world=2 on an idle pair. Three
repeats, `min` over 7 windows of 50 calls each:

| bytes | run 1 | run 2 | run 3 | GB/s |
|---:|---:|---:|---:|---:|
| 20,480 | 21.4 | 20.7 | **20.6** | 1.0 |
| 163,840 | 22.6 | 21.2 | 21.4 | 7.6 |
| 1,310,720 | 22.0 | 21.8 | 21.4 | 61.1 |
| 10,485,760 | 53.8 | 54.8 | 55.0 | 190.5 |

µs per all-reduce, cards 0 and 1, both idle at 0 MiB before each run.

**The cited figure is right.** 20.6–22.6 µs across the whole small-message range against
a cited 21.5 — inside 5%, and the spread between repeats at one size is under 1.4 µs. No
table built on 21.5 needs revising.

**"Flat 20 KB → 1.3 MB" is confirmed and its edge is located.** 20.6 at 20 KB against
21.4 at 1.3 MB is a **64x size increase for 4% more time**, so a collective in that band
is pure latency and message size is not a lever. Flatness ends above it: 10 MB costs
53.8–55.0 µs at 190 GB/s, where bandwidth takes over.

**The 108 µs outlier was contention, and the archive proves it without a new run.** The
Aug 29 world=6 logs contain two anomalous cells — 108.0 µs at 163,840 bytes in
`nccl6b.log`, 69.1 at 20,480 in `nccl6.log` — each 3–5x its neighbours. The same sizes in
the sibling runs read 22.4 / 22.6 / 22.9. Cards 2-7 were running another team's job
throughout. The probe already anticipates this, taking `min` over 7 windows because
"small-message rows jitter 3-5x from other tenants" (`nccl_probe.py:31`); on an idle pair
the jitter is gone entirely.

For the record, the archived runs (`/work/nccl.log` world=8; `/work/nccl6{,b,c}.log`
world=6 on cards 1-6):

| bytes | w=8 | w=6 a | w=6 b | w=6 c |
|---:|---:|---:|---:|---:|
| 20,480 | 24.2 | **69.1** | 23.0 | — |
| 163,840 | 22.6 | 22.9 | **108.0** | 22.4 |
| 1,310,720 | 32.3 | 25.2 | 25.6 | 24.1 |
| 10,485,760 | 89.8 | 87.6 | 124.5 | 87.7 |

Gaps are the logs' own — each was captured with `tail -8`.

## Rule

Provenance has two ends and the tree is only one of them. The open entry's search was
correct and its conclusion — "no run of it is recorded anywhere" — was false, because the
recording sat on the machine the run happened on. A measurement that never comes back
into the repository is invisible to exactly the search that would justify re-running it.

Second half: **one anomalous cell is not a finding until a repeat says so.** Had only
`nccl6b.log` survived, the honest report would have been an unexplained 4.7x spike inside
the region three cost tables call flat, and the CP scheme choice would have gone back
under review over a busy neighbour.

## Results

| date | commit | machine | target | world | 20 KB | 163 KB | 1.3 MB | 10 MB |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-07 | 6efb1ca | H20 cards 0+1 | cuda | 2 | **20.6** | 21.2 | 21.4 | 53.8 |
| 2026-08-29 | unrecorded | H20, 8 cards | cuda | 8 | 24.2 | 22.6 | 32.3 | 89.8 |
| 2026-08-29 | unrecorded | H20, cards 1-6 | cuda | 6 | 23.0 | 22.4 | 24.1 | 87.6 |

µs per all-reduce, best of the repeats per size, which is what `min`-over-windows already
does within a run. The Aug 29 logs carry no commit stamp — they predate `.synced_commit` —
so their provenance is the date and the launch line only.

Raw artifacts: `/work/nccl2.log`, `/work/nccl2r2.log`, `/work/nccl2r3.log` (this run);
`/work/nccl.log`, `/work/nccl6.log`, `/work/nccl6b.log`, `/work/nccl6c.log` (Aug 29).
`scripts/nccl_probe.py`, unchanged — it is world-agnostic, and the only reason it had
never run at world=2 is that its docstring says `--nproc_per_node=8`.

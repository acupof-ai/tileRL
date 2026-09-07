# The DRAM tier is 3.57x warm, 2.1x averaged, on the fixed publisher — H20, 2026-09-08

> Status: Shipped (measurement only; no default flips)

## Context

[The tier's 3.57x](2026-09-07-the-dram-tier-is-357x-when-the-budget-is-pressured.md) was measured
against a publisher emitting 62 prefix entries per 31k-token miss. That publisher is fixed as of
a43a379 ([cut the prefill publish flood](2026-09-08-cut-the-prefill-publish-flood.md)): 2 publishes
per row at any prompt length. The open question was whether the tier still pays once the flood it
was absorbing is gone, since a fix that relieves the pressure could leave the tier nothing to buy.

Two arms, one flag apart, on card 0 of the H20 pod. **Not the 09-07 cell** — see the parameter
table below, which is the part of this entry most likely to be misread.

Workload: `scripts/bench_chat_interleaved.py --sessions 12 --turns 4 --grow 40 --sys-tokens 30000
--ttft`. Twelve concurrent conversations opening with the same ~30k system prefix, then diverging;
prompts run 31.2k → 43.4k over 48 rows. `--state-bytes 1073741824` (~6 resident snapshots at
157 MiB), `--model qwen38-27b` NVFP4, engine flags left at their defaults (`blocks` resolved to
48099). Tree `/work/tilerl-s-v100-sm70-fp4`, `.synced_commit` a43a379, echoed by `pod_run`'s first
log line in both arms.

## What Worked

| | tier off | tier on (`--dram-bytes 8 GiB`) | ratio |
|---|---:|---:|---:|
| total wall | 624.38 s | **281.16 s** | **2.22x** |
| mean TTFT | 12.64 s | 5.42 s | 2.33x |
| prefix evictions | 139 | **0** | — |
| dram promotions / demotions | 0 / 0 | 36 / 175 | tier engaged |
| prefix hits | 47 | 47 | identical |
| prefix published | 190 | 190 | identical |
| compiles | 0, `turns_with_compiles []` | 0, `turns_with_compiles []` | — |

`tokens_generated` is pinned at 1536 in both arms and mean TTFT is 12.64 of a 13.01 s mean turn,
so this cell is prefill-bound and **tok/s (2.46 → 5.46) is a TTFT reciprocal, not an independent
axis.** The comparison is wall clock, TTFT and evictions.

## The mechanism is in the per-turn split, not the total

| turn | tier off | tier on | ratio | promotions (on) |
|---:|---:|---:|---:|---:|
| 0 | 184.59 s | 157.92 s | 1.17x | 0 |
| 1 | 157.97 s | 42.20 s | 3.74x | 12 |
| 2 | 138.69 s | 36.27 s | 3.82x | 12 |
| 3 | 143.13 s | 44.77 s | 3.20x | 12 |
| all | 624.38 s | 281.16 s | 2.22x | 36 |

**A coincidence to defuse before someone reads it as confirmation:** turns 1-3 pooled give
157.97+138.69+143.13 over 42.20+36.27+44.77 = **3.57x**, digit-for-digit the 09-07 headline. It is
not that number. The 09-07 figure is a whole-cell ratio at different parameters against the old
publisher; this is a warm-turns-only subset of a different cell against the fixed one. Two unrelated
quantities landing on the same two digits is the kind of agreement that gets cited as a replication,
so it is written down here as arithmetic rather than left to be discovered.

Turn 0 is roughly equal because nothing has been promoted yet — there is nothing to serve from, and
the residual 1.17x is a startup transient in two rows of the off arm, dissected below. From turn 1
the tier-on turns fall to about a quarter, at exactly 12 promotions per turn, one per session.

**Turn 1 tier-on is bimodal and a mean hides it:**

```
2.02 2.03 2.03 2.07 2.27 2.27 2.28 2.31 2.31 2.33 10.12 10.16
```

Ten sessions served from DRAM at ~2.2 s; two paying full prefill at ~10.1 s, because they arrive
before their prefix is promoted. The turn mean is 3.52 s, which describes neither population. Tier
off's same turn is flat 10.75–20.76 s with no second mode. `prefix_evictions 139 → 0` is the same
fact from the store's side: with the tier the budget stops dropping entries altogether.

## This is not the 3.57x cell, and the difference is the workload

Recorded because it was asserted twice before the 09-07 entry's Context was read, and a peer
endorsed the claim on that basis:

| parameter | 09-07 cell | this cell |
|---|---|---|
| `--turns` | 3 | 4 |
| `--grow` | 10 | 40 |
| `--max-batch` / `--max-ctx` | 1 / 40960 | unset |
| `--blocks` / `--slots` | 8192 / 16 | unset (48099 resolved) |
| prompt range | ~31k | 31.2k → 43.4k |
| rows | 36 | 48 |

`--grow 40` over 4 turns is why prompts reach 43.4k. So 624.38 s here against the 09-07 cell-1's
199.35 s is **the workload being larger, not a regression**, and the 2.22x above licenses nothing
about 3.57x. A separate pair at the 09-07 parameters, both arms post-fix, is what settles that.

## Turn 0 should be a tie, and it is — once two startup rows come out

Turn 0 promotes nothing, so both arms do identical work and the 1.17x above is 26.67 s the tier
cannot explain. A reviewer flagged it as either ~14% run-to-run variance on this cell (which would
put the 3.20–3.82x spread across turns 1–3 inside the noise) or a second mechanism such as
`--dram-bytes` changing pool geometry at startup. Read row by row in arrival order, it is neither:

| turn-0 rows | tier off | tier on | gap |
|---|---:|---:|---:|
| conv A + B (first to arrive) | 58.36 s | 28.70 s | **29.66 s** |
| conv C…L (the other ten) | 126.23 s | 129.22 s | **−2.99 s** |
| median row | 12.83 s | 13.11 s | −0.28 s |

Rows C–L are a tie with the **tier-off arm 2.4% faster**, and its median row is lower. The entire
turn-0 gap is two rows: off's conv A at 36.84 s and B at 21.52 s, against a tier-on arm whose worst
row is 14.90 s. A first-arrival transient in one arm — not run-to-run variance, and not
`--dram-bytes` changing startup geometry, which would move all twelve rows rather than two.

**Correcting for it lowers the headline, not raises it.** The transient is in the *off* arm, so it
inflates the numerator: removing it makes off/on smaller. Replacing each arm's turn 0 with twelve
times its own C–L mean (off 12.62 s, on 12.92 s):

| | off total | on total | ratio |
|---|---:|---:|---:|
| as measured | 624.38 s | 281.16 s | **2.221x** |
| off arm de-transiented | 591.27 s | 281.16 s | 2.103x |
| both arms de-transiented | 591.27 s | 278.30 s | **2.125x** |

So the honest figure is **2.10–2.13x, and the reported 2.221x is about 4.5% too high** — the
transient is a windfall to the tier's apparent advantage, not a tax on it. An earlier draft of this
entry claimed the opposite ("2.22x is an underestimate"), which was a sign error in the direction
that flattered the result; a reviewer caught it.

**Two smaller readings.** The variance hypothesis is refuted for the body of the cell: ten paired
rows agreeing to 2.4% means the turns 1–3 ratios of 3.20–3.82x are not sitting inside a ±14% band.
And the C–L "tie" is not quite a tie in the tier's favour — with the tier on and promotions active,
the ten steady turn-0 rows are **2.4% slower**, a small standing cost of having the tier enabled that
is visible only while there is nothing yet to promote.

## The ratio depends on how many sessions lose the promotion race

Turn 1's two slow rows contribute 20.28 s of that turn's 42.20 s — **48% of the turn from 17% of the
sessions**. If arrival ordering is arbitrary, that count is a per-run variable rather than a constant
of the tier: a run where four sessions lost the race would report a materially lower ratio with
nothing having changed. The honest form is therefore **2.22x with 2 of 12 sessions losing the race at
turn 1**, and anyone comparing against another run must check that count before comparing ratios.
Each arm was run once; the count is not established as stable.

## Rule

**The tier is ~3.6x once warm, ~1x on the first turn, and ~1x for any session that races promotion.**
2.1–2.2x is the average over a cell containing all three regimes, and it is the least informative way
to state the result: turns 1–3 run 3.20–3.82x (3.57x pooled), turn 0 runs at parity with nothing
promoted, and within turn 1 the two sessions arriving before their prefix is promoted pay the full
10.1 s. A cell measured over one turn reports the tier as worthless; a cell quoted as a single mean
hides that its benefit is gated on promotion having already happened.

Corollary that survives the publisher fix: **a tier that absorbs pressure still pays after the
pressure's source is reduced.** Publishes went from 62 per 31k miss to 2 per row, and the remaining
pressure at a 6-snapshot budget still evicts 139 entries without the tier and 0 with it.

And: **compare arms on the axis the cell is bound by.** This one is prefill-bound, so tok/s is the
TTFT reciprocal and moves by construction.

## Results

| date | commit | machine | target | model | mean TTFT s | mean turn s | total wall s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-08 | a43a379 | H20 card 0 | cuda | qwen38-27b NVFP4 | 12.64 | 13.01 | 624.38 |
| 2026-09-08 | a43a379 | H20 card 0 | cuda | qwen38-27b NVFP4 (+dram 8 GiB) | 5.42 | 5.86 | **281.16** |

Raw artifacts: `/work/pub271off.out`, `/work/pub271on.out`, server logs
`/work/pub271{off,on}-serve.log`, wrapper logs `/work/pod_run_pub271{off,on}.out`.

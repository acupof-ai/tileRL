# The DRAM tier is 3.57x on wall clock once the state budget is actually pressured — H20, 2026-09-07

> Status: Shipped (measurement); the DRAM default does NOT flip on this card — see Deployment

## Context

The DRAM and SSD snapshot tiers exist for one condition: **concurrent sessions >
the snapshots HBM keeps resident**. Every prior verdict on them swept one axis
and read `0` promotions, because the other axis had no flag: the budget came only
from `engine.py:1606`, a quarter of free memory.

`--state-bytes` (f038c96) makes the numerator settable, so this is the first
measurement of the tier inside its own regime rather than outside it.

Workload: `scripts/bench_chat_interleaved.py --sessions 12 --turns 3 --grow 10
--sys-tokens 30000 --ttft`. Twelve concurrent conversations, each opening with the
same ~30k-token system prefix (tool schemas plus standing rules, the shape a
Claude Code turn resends unchanged), then diverging. One `serve` per cell,
`--max-batch 1 --max-ctx 40960 --blocks 8192 --slots 16`, H20 card 0, real
`qwen38-27b` NVFP4, tree `/work/tilerl-s-tierbench` sha `169d7bd`.

## What Worked

| cell | state budget | resident snapshots | tier | wall clock | vs cell 1 |
|---|---|---:|---|---:|---:|
| 1 | 1.0 GiB | 6 | off | 199.35 s | 1.00x |
| 2 | 1.0 GiB | 6 | dram 8 GiB | **55.82 s** | **3.57x** |
| 3 | 1.0 GiB | 6 | dram + ssd | 103.00 s | 1.94x |
| 4 | 2.0 GiB | 13 | off (control) | 59.38 s | 3.36x |

All four report `compiles: clean` — zero JIT inside any measured turn, read from
each server's own log.

**The regime is reached, and the capacity confirms it independently.** The store's
own `prefix_entries_capacity` reads **6** at 1.0 GiB and **13** at 2.0 GiB,
matching the budget arithmetic rather than restating it, and
`prefix_state_bytes / prefix_entries` = 784465920 / 5 = **156.9 MiB** per
snapshot, which is the 157 MiB the derivation assumed.

**Cell 1 is the pressured arm and it fails at turn 2**, per-turn TTFT over the 12
sessions:

| turn | cell 1 (off) | cell 2 (dram) | cell 4 (control) |
|---|---|---|---|
| 0 | 0.41–14.12 s, 11/12 hits | 0.41–18.51 s, 11/12 | 0.41–14.06 s, 11/12 |
| 1 | 0.84–0.98 s, 12/12 hits | 0.64–0.82 s, 12/12 | 0.83–1.00 s, 12/12 |
| 2 | **1.40–14.25 s, 1/12 hits** | 0.81–0.95 s, **12/12** | 1.36–1.56 s, 12/12 |

Cell 1's turn 2 loses 11 of 12 hits and pays 14.1 s of prefill for each, with
`prefix_evictions` totalling **835** across the run. Cell 2 has **0** evictions
and 12/12 hits: the 180 demotions moved the snapshots to host memory instead of
dropping them, and 24 promotions brought back what turn 2 asked for. Same code,
same workload, same budget — the tier is the only difference.

**The control settles that it is the budget and not the workload.** At 2.0 GiB the
tier-off arm gets 12/12 hits at turn 2 with 190 evictions and 59.38 s. So cell 1's
199 s is what a 6-snapshot budget does to 12 sessions, not what 12 sessions cost.

**Cell 2 beats the unpressured control**, 55.82 s vs 59.38 s = 1.06x. The tier
does not merely recover the pressure it was given; a demoted-and-promoted snapshot
costs less than the control's own eviction-and-reprefill, which the control still
pays 190 times.

## The SSD tier served nothing and cost 1.85x

Cell 3's hit, promotion and demotion counts are **byte-identical** to cell 2's
(35 / 24 / 180) — so every snapshot cell 3 served came from the DRAM tier, and the
SSD layer changed no outcome. Its own counters say so directly:

    ssd_hits: 0        ssd_recovered: 0     ssd_faults: 0
    ssd_saves: 86      ssd_save_ms: 64768   ssd_bytes: 19492071402
    ssd_offered: 108   ssd_refusals: 33     ssd_evictions: 34

**0 hits, 0 recovered, and 64.8 s inside `ssd_save_ms`** for 19.5 GB written to
`/dev/vda2`. The saves also sit in the demote path, which is visible as a second
cost: `dram_demote_ms` goes 4883 → **8428** and `dram_promote_ms` 2 → **95** for
the identical 180 demotions and 24 promotions. Adding a layer under DRAM slowed
the layer above it.

This reproduces the serve-path finding
([errors/2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md](../errors/2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md))
at a different budget and with the mechanism now attributed to `ssd_save_ms`
rather than inferred from wall clock.

## Pending: the publisher this was measured against has changed

The 3.57x was measured against a publisher emitting **62 entries per 31k-token miss**, and this
entry's own mechanism paragraph says the tier does not stop that flood — it gives the flood
somewhere to go. That publisher is fixed as of
[wins/2026-09-08-cut-the-prefill-publish-flood.md](2026-09-08-cut-the-prefill-publish-flood.md):
2 publishes per row at any prompt length. **So 3.57x is provisional** until the same cell runs on
the fixed publisher, and this section is the flag rather than a revision — a token-count probe on
CPU cannot rewrite a wall-clock verdict, and the card cell is scoped and pending.

One structural fact from that probe does land here, because it is not a speed claim: **the tier is
inert below a 4-snapshot budget.** At `--dram-bytes` worth 3 snapshots it demoted nothing and
changed no cell of a 9-cell grid, under both the flooding and the fixed publisher. It is
all-or-nothing at these sizes, so a budget that cannot hold ~4 snapshots is not worth wiring at
all.

## Deployment: the default does not flip

**The mechanism verdict and the deployment verdict rest on different evidence and
must not be merged.** The 3.57x above is measured, and it licenses the tier for a
pressured card. Whether an H20 reaches that pressure is derived arithmetic, not a
measurement: `mem_get_info()[0] // 4` there is 17.9 GiB, which at 156.9 MiB is
**116 resident snapshots**, so DRAM pressure begins near **115 concurrent agent
sessions**. No benchable load reaches that, which is exactly why this cell needed
`--state-bytes` to exist.

So on an H20 at its shipped default, `--dram-bytes` should stay off; nothing here
changes that. The card where the shipped default IS pressured is the V100, and the
figure is read off the live child rather than derived: `/health` on pid 3128149
(`--max-ctx 32768 --max-batch 1`, depth 1) reports
`prefix_state_bytes_budget: 1845067776` = **1.718 GiB** and
`prefix_entries_capacity: 11`, with `prefix_state_bytes / prefix_entries` =
313786368 / 2 = **149.6 MiB** per snapshot. So 11 snapshots against 12 sessions —
pressured, but by one session rather than by three.

**An earlier draft of this entry said 9 snapshots from 8 GiB at 144 MiB, and both
operands were wrong.** The 8 GiB is `PrefixStore`'s own default, not what this card
runs: `build_engine` passes `mem_get_info()[0] // 4` and that quarter is taken
*after* weights and pools, so on a 32 GB V100 with this config it is 1.718 GiB.
The 144 MiB is the bf16 snapshot size; this checkpoint's is 149.6 MiB. Two errors
that happened to compose into a plausible 9. The V100 grid is a measurement to
run, not arithmetic to publish, and there the tier is HBM→SSD with no host layer
(`kv_cache.py:390-396`) — the arm this entry shows to be the worst of the three.

**A block costs 1.0 MiB here and 2.0 MiB there, and the difference is the dtype.**
`PagedKvPool` allocates `k_pool` and `v_pool` as two tensors of
`[planes, blocks, heads, 16, head_dim]`, and the engine passes `Backend.io`:
`backend.py:359` is `float32 if arch in ("cpu", "metal", "sm70") else bfloat16`. So one
block is 1.0 MiB for K+V at bf16 on this H20 and **2.0 MiB at f32 on the V100**, both
measured on a real pool at the 27B's shape. The 12x31k agent shape therefore needs
24 GiB of blocks on the H20 and **48 GiB on the V100** — which is why it does not fit a
32 GB card, and why a byte-size claim about this pool is wrong without its dtype.

## Rule

**A tier's own regime is a flag on both of its terms, and the verdict inside it
does not transfer to a card that never enters it.** 3.57x is real and 116
snapshots is real; the second one is why the first does not flip a default.

**A tier that changes no counter changed no outcome.** Cell 3's hits, promotions
and demotions are identical to cell 2's to the digit, so `ssd_hits: 0` is not a
weak signal to be argued past — the layer is inert and its 64.8 s of saves are
pure cost. Read the counters of the layer under test before reading its wall clock.

## Results

| date | commit | machine | target | model | cell | wall clock |
|---|---|---|---|---|---|---:|
| 2026-09-07 | 169d7bd | H20 card 0 | cuda | qwen38-27b NVFP4 | 12 sess, 1.0 GiB, off | 199.35 s |
| 2026-09-07 | 169d7bd | H20 card 0 | cuda | qwen38-27b NVFP4 | 12 sess, 1.0 GiB, dram | 55.82 s |
| 2026-09-07 | 169d7bd | H20 card 0 | cuda | qwen38-27b NVFP4 | 12 sess, 1.0 GiB, dram+ssd | 103.00 s |
| 2026-09-07 | 169d7bd | H20 card 0 | cuda | qwen38-27b NVFP4 | 12 sess, 2.0 GiB, off | 59.38 s |

Raw artifacts: `/work/tb1w.log`, `/work/tb2w.log`, `/work/tb3.log`,
`/work/tb4.log` on the H20, each carrying its own 36 rows, per-session totals and
`final_stats`.

## A cold cache costs 3.8 s per compile, and a neighbour row is not the control

Cell 1 was first run against a cold `/work/tilelang_cache` and came out at
250.14 s with 12 compiles inside measured turns. It was re-run warm rather than
reported, because the later cells would inherit the warmed cache and be
incomparable.

The re-run also refuted the obvious way to price those compiles. Subtracting a
clean same-turn *neighbour* row gave 3.82 / 3.87 / 3.85 s per compile, agreeing
within 1.3% — and one of the four estimates was still wrong. Against the same row
re-run warm:

| dirty row | cold | warm, same row | per compile | neighbour estimate |
|---|---:|---:|---:|---:|
| turn 0 conv A (6 compiles) | 42.58 s | 14.49 s | 4.68 s | 6.95 s ✗ |
| turn 0 conv B (2) | 8.51 s | 0.92 s | 3.80 s | 3.82 s |
| turn 1 conv A (2) | 8.97 s | 1.35 s | 3.81 s | 3.87 s |
| turn 2 conv K (2) | 22.11 s | 14.61 s | 3.75 s | 3.85 s |

Turn 0 conv A is a prefix **miss** (`hits=0`, `evict=55`) and its neighbour was a
hit, so the neighbour differed from it by 13.6 s of prefill before any compile.
Three mutually consistent estimates did not detect that; the same row re-run did.
Totals agree independently: 250.14 − 199.35 = 50.79 s, against 50.80 s summed from
the four same-row deltas.

**Consistency among estimates that share a premise tests the estimates, not the
premise.** All three agreeing rows carried exactly 2 compiles, so their agreement
measured only that per-compile cost is stable. The counterfactual has to come from
the same row, not a neighbouring one. (Raised by tilerl-48 before the warm run
landed, and it changed the number.)

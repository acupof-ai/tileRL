# Sparse host cold prefix tier: flat RSS and free at 12 sessions — 2026-09-13

## Context

The standing serve-path reject was [the SSD tier 1.65x worse per turn at 12
sessions with zero hits](../errors/2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md).
That entry measured the **dense `KvTier`** (`--ssd-path`, `torch.save` per
prefix), a code path none of the recent work touched. This is the matching
measurement of the **new sparse host cold tier** (`HostKvPages` +
`SparsePrefixCache`, #556/#562) under the same multi-session interleaved
workload — the class the 256k single-stream leak fix actually lived in.

Sparse is opt-in and fidelity-flagged (sm90 B>1 continuity defect open, #563);
this is a **tier-perf / memory** result, not a sparse-fidelity verdict.

## Workload

`scripts/bench_tier_wall_clock.py --cold-spill`, card 3 H20, head 734d5a70
(#562 merged). 12 conversations interleaved A1 B1 … L1 A2 …, 3 growing turns
(`--grow 40`, prompts ~8.4k tokens by turn 3), `--max-ctx 49152 --slots 3`,
both arms `--sparse-k 128`; the ON arm adds a 6 GiB pinned host budget plus a
12 GiB mmap cold spill. One fresh server per arm, a full-shape (12×3) warm-up
run first so `compiles == 0` in every measured row. Off arm = sparse without
the SSD spill flags (host tier auto-sized to the working set).

## Measured (36 turns per arm)

| | off (sparse, no SSD) | on (6 GiB host + 12 GiB spill) |
|---|---:|---:|
| per-turn mean | 4.019 s | **3.971 s** |
| per-turn median | 3.476 s | 3.474 s |
| on/off ratio | — | **0.988** |
| server RSS first → max | 1.99 → 6.88 GiB | 1.99 → **6.88 GiB (flat)** |
| interleaved prefix hits | — | 24 |
| cold demotions / promotions | — | 8268 / 156 |

The #556 leak does not recur under multi-session churn: RSS climbs to the
shared-prefix working set (plateau 1.15 GiB in the cold tier, 6.88 GiB total
including weights' runtime footprint) and stays there across all 36 on-turns,
and the ON arm is indistinguishable from OFF per turn (0.988x). Interleaved
conversations actually adopt published prefixes (24 hits) — the new tier has a
credit side, unlike the dense `KvTier`'s 0.

## Limitation — the mmap spill file is not exercised at this size

`cold_ssd_bytes` was **0.0** in the 6 GiB run: the ~8.4k-token prompts built
only ~1.15 GiB of shared prefix, under the 6 GiB host budget, so nothing spilled
to disk. That row proves the host cold tier is leak-free and free at 12
sessions, but not the spill file under churn. A second arm pinned the host
budget to **256 MiB**, forcing most of the shared set to spill:

| host budget | on/off ratio | RSS first→max | spilled (private + prefix) | hits |
|---:|---:|---:|---:|---:|
| 6 GiB | 0.988 | 1.99 → 6.88 GiB | 0 B | 24 |
| 256 MiB | 0.993 | 1.95 → **5.99 GiB** | **195 MiB + 1.069 GiB** | 24 |

With the file actually written (1.069 GiB prefix spill on disk), RSS stays
flat ~0.9 GiB lower and the spill arm is still 0.993x per turn — spilling is
free on the wall clock and memory stays bounded. The 256 MiB off baseline was
4.146/3.633 s (its own same-run off server); both arms compiles 0. The 256k
single-stream run separately stressed the file alone (11 GiB spilled, RSS
pinned at the 6.0 GiB budget, cumulative write 1.1% of prefill).

## Relationship to the 1.65x entry

That error stays **open and stands as written**: it concerns the dense
`KvTier` (`--ssd-path`), which is unchanged and still publish-but-never-serve
in the dense default. The two are different classes behind different flags;
this win does not flip that reject.

## Rule

Re-running a verdict must re-run the CODE CLASS the verdict named. A tempting
"same `--ssd-path` flag" rerun here would have measured `KvTier`, which no fix
touched; the repaired path is reached only through `--sparse-k` +
`--cold-ssd-path`. And a churn tier is judged on RSS monotonicity and a real
credit side (hits), not only per-turn latency — a flat tier that serves
nothing would still be the 09-06 outcome.

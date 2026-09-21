# A full cold tier makes sm70 sparse 32k decode long-tail — 2026-09-17

> Status: open, measured. Not fixed. Surfaced during the draft read-window
> W=2048 sign-off (#698) but independent of it. Related: the build-time VRAM
> floor / shrink-realloc OOM work (#654/#656), and the #665 on-line sm70 sparse
> tick timing split (per-stage numbers in the #698 bench data
> wins/draft-window-2048-2026-09-17/decision-note.md).

>
> **Update 2026-09-19:** in the current 1 GiB-RAM/8 GiB-SSD tier the steady
> finalize tail is gone (1–2 ms) and the long tail moved to a once-per-request
> cold SSD close (~3.1 s); see
> [2026-09-19-sm70-32k-decode-physical-wall-and-ssd-close-tail.md](2026-09-19-sm70-32k-decode-physical-wall-and-ssd-close-tail.md).
>
> **Note 2026-09-21:** the “cold SSD close ~3.1 s” path cited here is deleted — the refactor publishes a page once when it leaves the pool and close moves no KV bytes (Epic #779 M3, #787/#789; see [errors/2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md](2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md)). The measurements above are unaffected; only the pointer to a now-removed mechanism is stale.

## Context

V100 sm70 hybrid serve, main `07eca439`, draft read window W=2048, flags
`--kv-cold-bytes 8589934592 --cold-ssd-bytes 8589934592`, sparse k=128,
d1, sparse-min-tokens 8192, `TILERL_STEP_TIMING=1`. The 32 GB card holds
~31.3–31.6 GB live; idle `device_free` is only ~200–480 MB.

A primed (warm) 32k sparse request decodes far below the ~9 tok/s the steady
tick predicts: 1.4–2.95 tok/s with the cold tier full, **7.86 tok/s** on a
fresh boot with an EMPTY cold tier. The median decode tick is healthy in both
states (~175–190 ms); the entire loss is a heavy long tail whose frequency
tracks cold-tier fullness.

Within one warm 32k run, consecutive ticks at the same `cmax=1916 own_w=8`
split bimodally: `model=157–168 ms` on most ticks, `model=1169–1259 ms` on
others. The same cmax bucket at two speeds rules out attention growing with
cmax, cmax-bucket recompile, sparse_select (1–10 ms), draft_step (7–45 ms
under W=2048), and offers publishing for the worst ticks.

## Root cause — two distinct long-tick mechanisms

**1. `sparse_finalize` batch page moves (dominant, scales with fullness).**
A tick that moves/drops pages pays `sparse_finalize = 629–667 ms` for
`offers_pages=129–144` (one smaller event 186 ms / 72 pages; empty-tier run
81 ms / 112 pages). The same ~120–144-page batch costs **7–8x more
(81 → 629 ms)** once the cold tier is full and device_free is ~200 MB. At
the edge of free memory, page alloc/free plus demotion thrash — the same
regime the build-time VRAM-floor work (#654/#656) identified. This term gets
strictly worse as the cold tier fills.

**2. "Hollow" forward ticks (unattributed, needs a per-tick probe).**
`total=1.1–1.4 s` with the forward envelope ~1.1 s but its measured inner sum
only ~0.3–0.5 s (`model` ~330–520 ms), `sparse_finalize` 1–3 ms,
`offers_pages=0`; three were consecutive in one run (model 1275/1303/1326
ms). Time is inside the model forward but uncovered by any current segment —
an implicit GPU sync or allocator stall. It occurs at both empty and full
tiers (weaker fullness correlation than mechanism 1) but always inside the
~200–480 MB device_free band. It has no counter yet; a per-tick device
alloc/free or sync probe (extending the #639 note) is required to attribute it.

`kv_cold_promotions` delta during a primed warm is **0** (a +23 seen in one
earlier window was cross-request): this is not cold-pool→device promotion
during decode.

Two-phase isolation (same boot, same W=2048; only cold-tier fill changed):

| phase | kv_cold_shared | device_free | warm 32k tok/s | tick p50 | p90 | max | >300 ms |
|---|---|---|---|---|---|---|---|
| 1 empty | 0 → 2.3 GB | 485 MB | **7.86** | 176 | 286 | 909 | 1/10 |
| 2 full | 8.07 GB | **206 MB** | **4.25** | 175 | 790 | 958 | 2/5 |

Two further primed samples at the full tier: C0 6.56 tok/s (p50 180, p90 846,
2/10 >300 ms), C1 2.69 tok/s (p50 393, p90 1337, 7/12 >300 ms). Across ~27
full-tier decode ticks the median stays ~175–190 ms while the >300 ms fraction
is 40–58 %, with repeated 1.1–1.4 s ticks. The 7–8x finalize regression
reproduced in three separate runs (P2, C0, C1); per-primed-warm n is only
5–12 ticks, so the rate needs a larger dedicated run.

## Why it is isolated from the W=2048 change and #695/#697

At the **same W=2048**, changing only cold-tier fill flips warm 32k between
7.86 (empty) and 2.69–6.56 (full) tok/s — cold-tier fullness is the controlled
variable. The healthy median (~175 ms, draft 7–13 ms) matches the modeled
served prediction at W=2048, and the long ticks carry `offers_pages=0` (the
draft window engages correctly, `windowed_seq_len=2048`): their time sits in
`sparse_finalize` and the unattributed forward wait, neither of which the draft
read window touches. A `TILERL_DRAFT_ATTN_WINDOW_TOKENS=0` control was
therefore judged unnecessary — fill level already isolates the cause, and a W
change cannot move page finalization. The W flip (#698) is green on its own
merits, and this is unrelated to the #695/#697 probes.

## Fix — direction only, open (nothing landed)

- **Reclaim headroom / bound concurrency**: revisit `--kv-cold-bytes` and the
  shrink-realloc floor from #654/#656 so finalize does not run at ~200 MB free,
  and/or cap 32k concurrency so the shared cold pool cannot reach the full-tier
  state during a serve. The intended effect is flattening the type-1 ~629 ms
  finalize batch (the dominant, fullness-scaled term). Not decided: re-sizing
  versus concurrency limiting is the open design choice.
  - **Landed, opt-in, default 0 — device-measured 2026-09-18: no-op for this
    tail.** Unlike the disproved reserve arms (the fraction fence does not shrink
    the arithmetic-sized sparse pool and then OOMs at pool build; host
    `kv-cold-bytes` does not raise device free), this builds the sparse main pool
    SMALLER and caps the greedy 2/3 draft fit (`sparse_pool_fit_headroom`), so the
    target MiB is free after both pools attach — reclaim headroom at the finalize
    edge rather than a post-hoc fence. CPU-only gates: a pure-arithmetic fit test
    (exact-max main count, draft-cap interaction, floor-conflict raise) and the
    off-cuda no-op/reporting wiring. Capacity cost on the 27B (sm70 f32, main
    block 2.0 MiB, per-slot hot ceiling 554 blocks): the main pool is
    1.08/2.16/4.32 GiB at 1/2/4 slots, and holding 768 MiB removes ≥384 main
    blocks (≥0.69 slot peaks), more where the greedy dense draft pool re-grows;
    the single-slot floor is the hard minimum, and an infeasible target raises at
    build naming the floor's residual. **The four-arm 0/384/768/0 run on V100
    (2026-09-18) returned no-op: every arm sizes the same 2213-block device pool
    and the fill `sparse_finalize` distribution is unchanged (p50 190–222 ms /
    max 729–1138 ms). The solver prices device pools only, so it never sees the
    runtime host cold tier this tail actually lives in** — see
    [wins/2026-09-19-cold-tier-olru-zero-copy-spill.md](../wins/2026-09-19-cold-tier-olru-zero-copy-spill.md)
    and the
    [2026-09-19 sm70 32k wall verdict](2026-09-19-sm70-32k-decode-physical-wall-and-ssd-close-tail.md);
    the host tier's own O(n) LRU/spill cost was fixed in #735.
- **Attribute mechanism 2 first**: add a per-tick device alloc/free + sync probe
  so the hollow-forward ticks are localised before treating them as the same
  allocator root cause.
- Then a larger-n full-tier run to pin the >300 ms rate before/after.

## Rule

On a card run within a few hundred MB of full VRAM, a KV cold tier that fills
to capacity turns a healthy median tick into a long-tail serve: the steady-state
median can read fine while 40–58 % of ticks stall, dominated by 7–8x finalize
page-move batches and an unattributed forward wait. Judge capacity by the tail
under a **full** tier, not a freshly-booted empty one, and do not attribute the
regression to an unrelated decode change that the fill-level control already
isolates.
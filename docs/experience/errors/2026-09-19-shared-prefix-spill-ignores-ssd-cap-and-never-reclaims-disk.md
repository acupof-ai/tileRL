# Shared-prefix SSD spill ignores the cold-SSD byte cap and never returns disk — 2026-09-19

> Status: landed behind an env gate (`TILERL_COLD_PREFIX_SSD_CAP=1`), default
> OFF. Device partial confirmation on V100 2026-09-19 (659c2fbb): the physical
> `.prefix.bin` is pinned at 8192 MiB with zero drops under load (vendored
> size timeline); post-release trailing reclaim was NOT sampled in this window
> and stays pending-next-window. A full-cap steady state adds an evict/reload
> tail (one warm rep fell to 5.84 effective tok/s). Observed during the 2026-09-18
> headroom window: the `.prefix.bin` shared-prefix spill grew to 13.6 GiB
> logical / 25 GiB physical against an explicit `--cold-ssd-bytes` of 8 GiB.

## Context

Two distinct spill files sit behind one `--cold-ssd-bytes` setting:

- the **private** cold page spill (`sparse_cold.bin`), whose capacity is counted
  in admission (`HostKvPages.ssd_capacity_bytes`, `cold_capacity_blocks`);
- the **shared-prefix** publish cache (`sparse_cold.prefix.bin`), which
  `_write_shared_ssd` wrote with **no capacity check at all**. Every published
  prefix page was appended regardless of the cap. A published prefix is
  serve-only (never read back after restart), so this was pure disk growth, not
  retained state.

The #735 extent growth made the mapping cheap (one ftruncate+mmap per 64 slots)
but only **reused** freed slots through the LIFO free list; it never shrank the
file, so a release wave left physical disk at the high-water mark forever.

## Fix (env-gated, default unchanged)

`TILERL_COLD_PREFIX_SSD_CAP=1` enables two behaviors on the shared spill only;
the private spill and every default (gate unset) path is unchanged:

1. **Capacity admission.** `_write_shared_ssd` refuses a page that would push
   `_shared_ssd_bytes` past `ssd_capacity_bytes` (the same `--cold-ssd-bytes`
   figure the private spill reports). It returns False, which is the existing
   "shared page stays in host RAM / spill disabled" safety path — a published
   cache may be forgotten but never raises or wedges a tick. The shared page
   remains readable from RAM; a later follower misses the prefix rather than
   reading a corrupt/oversized file.
2. **Trailing-extent reclaim.** `ColdSsdFile(reclaim=True)` tracks live slots
   per extent; `forget` truncates the file back when every extent at the
   high-water end is empty (one ftruncate + remap), so a release wave returns
   physical disk. Interior free extents still cycle through the LIFO free list.

Tail-collapse is the safe, portable reclaim: it uses ftruncate only (no
platform punch-hole ioctl). Mid-file fragmentation is left as a follow-up
(FALLOC_FL_PUNCH_HOLE / F_PUNCHHOLE) if physical bytes ever exceed the cap with
a non-trailing free pattern. `stats()` adds `kv_cold_shared_ssd_bounded`.

## Gates

CPU tier tests (hermetic, no device):

- bounded mode refuses the page past an N-page explicit cap, keeps it in RAM,
  and the counter stays at the cap; with the gate unset the same write is
  allowed (default behavior pinned);
- reclaim truncates on a freed trailing extent, leaves the file untouched when
  only an interior extent frees, collapses to the header after the last live
  slot, and surviving extent-0 pages still round-trip byte-exact; with
  reclaim=False forgetting never shrinks (default pinned).

## Device confirmation (V100 sm70, 2026-09-19, 659c2fbb)

cb2 = the cb1 batch gate plus `TILERL_COLD_PREFIX_SSD_CAP=1`, same
fill-2/warm-2 protocol (37.6k prompts, 1 GiB-RAM / 8 GiB-SSD f16). A 20-second
sampler recorded apparent + physical (`du`) `.prefix.bin` size and the health
logical counter through the run.

**Cap holds physically under load, no wrong pages (vendored).** Physical file
size sat at exactly **8192 MiB** at every full-tier sample and logical
`kv_cold_shared_ssd_bytes` at 8,589,279,232 B (~8.0 GiB); `kv_cold_drops=0`.
The uncapped arms reached 13.6 GiB logical / 16–25 GiB physical on the same
two-rep workload. The 60-sample physical-size timeline (`cb2-sizes.txt`,
20 s spacing) covers the fill + warm phase; its last sample (23:21:25) is still
on the 8192 MiB plateau.

**Reclaim NOT confirmed this window.** The sampler stopped with the serve still
at the cap and did not cover a post-release read, the test spills were deleted
on the restore to the flags-off production serve, and the cap path can't be
re-driven without a new window — so there is no on-disk shrink evidence here.
The earlier oral "8.6→8.1 GiB after release" was a unit muddle
(8,606,715,904 B = 8.016 GiB = the same 8192 MiB plateau in decimal GB, not a
later smaller value) and is withdrawn. Trailing-extent truncation is covered by
the hermetic CPU tier test; physical post-release reclaim stays
**pending-next-window** — re-sample `du` after the publish refs release with the
spill left in place.

**Follower correctness: client-terminal observation, not vendored.** A follower
repeating an identical 32k prompt returned byte-identical tokens
(`finish_reason=length`, `/health` `prefix_hits` delta +1) in the client-side
`follower_smoke.py` run on both cb1 and cb2. That script's stdout was not
redirected to a saved log and the boot log carries no per-request prefix-hit
line, so this is an unaudited client-terminal read, not a vendored artifact;
the prefix-hit assertion should be captured to a file in the next window.
`kv_cold_drops=0` (health at the time) is the one machine-readable no-wrong-page
signal that was read directly.

**New tail: full-cap evict/reload churn.** With the file at the cap during the
second warm rep, eviction-and-reload dominated: that rep's effective tok/s fell
to **5.84** (9.25 while the tier was under the cap in rep0), warm tick p90 rose
to 738 ms (from ~209 under cap-free arms), `pub_share_hold` hit a 5903 ms max
and `release_close_request` a 7651 ms max (11 tail ticks vs 6–7). Steady median
was unchanged (model 168 / tick 185 ms) — the cost is the cold-full tail, not
the trunk, and it is a new shape introduced by bounding a publish-only cache
that the current workload fills but rarely serves. The cap protects disk
correctness; it does not remove the close byte cost (the second PR's scope).

Vendored: `wins/close-batch-cap-device-2026-09-19/cb2.json`,
`close-segments.txt` and the full `cb2-sizes.txt` physical-size timeline; the
raw per-tick log is `~/tilerl-logs/serve-cb2.boot` on the box.

Both assertions were mutation-verified red with the cap test and the shrink
call independently disabled. Full CPU suite 1008 passed.

## Rule

A publish-only cache file does not inherit the capacity of the state spill
beside it — admission has to be enforced on every spill path, not assumed from
a shared flag. Extent growth needs a matching trailing-extent collapse, or the
LIFO slot reuse hides unbounded physical growth behind a bounded live slot
count.

## Results

| date | commit | machine | target | model | physical spill | drops / correctness | full-cap tail |
|---|---|---|---|---|---:|---|---|
| 2026-09-19 | pending PR | CPU (hermetic) | HostKvPages shared-prefix spill cap + reclaim | — | — | bounded cap enforced; trailing extents truncated to header | — |
| 2026-09-19 | 659c2fbb | V100 sm70 | shared `.prefix.bin` cap (filler+warm only) | Qwen3.8-27B-NVFP4, 37.6k sparse, 1G/8G f16 | pinned 8192 MiB under load (was 16–25 GiB); post-release reclaim NOT sampled, pending-next-window | 0 drops (health); follower identical = client-terminal only | rep2 eff 9.25→5.84 tok/s, tick p90 738 ms, close max 7.65 s |

Raw artifacts: tests `tests/test_sparse_kv_tier.py`; change `src/tilerl/kv_tiers.py`.

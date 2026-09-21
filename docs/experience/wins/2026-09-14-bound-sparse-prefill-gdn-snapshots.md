# Bound the host GDN snapshots a long sparse prefill accumulates — 2026-09-14

> Status: closed. CPU count gate red on main / green after; e2e prefix-hit
> follower token-equal. Device RSS confirmation pending the next V100 window
> (`pending-remote`, command below).

## Context

The V100 sparse 256k prefill was OOM-killed with zero SSD spill
([the kill record](2026-09-13-v100-256k-sparse-prefill-host-oom.md)). The
hypothesis was own-span K/V pages pinning into the 10 GiB host tier faster than
prefill demoted them. That hypothesis was wrong: `HostKvPages` pins K/V blobs
under its byte budget and LRU-spills past it, so K/V alone cannot grow to the
27.9 GiB RSS measured at 64k.

The bytes were a second container. Every aligned prefill chunk (512 tokens on
the serve path) ends on a whole-page boundary, and `_sparse_finalize` calls
`SparsePrefixCache.note_boundary`, which copies the GDN recurrent state to the
host for a possible prefix entry — **149.6 MiB per snapshot** (144 MiB GDN
state across 48 layers + 5.6 MiB conv window, f32) — into a bare per-request
dict, with no byte budget. The ponytail on the code named this exactly:
"retained per request without a byte budget."

During a single long prefill the dropped-page frontier stays at page 0: the
k+window+chunk hot pool keeps every early page of the one growing request
resident, so nothing drops and no frontier ever closes. The snapshots then
accumulate one per chunk for the whole prefill, entirely outside
`HostKvPages`' budget and outside its SSD spill path (which is why
`cold_ssd=0` at the kill).

| ctx | aligned boundaries | snapshot bytes |
|---:|---:|---:|
| 64k | 128 | 128 × 149.6 MiB = **18.7 GiB** |
| 256k | 512 | 512 × 149.6 MiB = **74.8 GiB** |

18.7 GiB of snapshots plus the bounded K/V cold tier, weights host mirrors and
allocator pinned staging reaches the measured 27.9 GiB peak on the 31 GiB box at
64k; 74.8 GiB at 256k is the SIGKILL, before any K/V page needs the spill file.

## What worked

`note_boundary` keeps at most **two** snapshots per request: the LOWEST
unconsumed boundary and the NEWEST.

- The frontier closure consumes boundaries in non-decreasing order. The lowest
  unconsumed snapshot is the one the next closure (`publish_dropped`) adopts.
- The newest is the prompt-end boundary, which freezes the entry a same-prompt
  follower adopts (`close_request` / the `at_prompt_end` freeze).

> **Note 2026-09-21:** `close_request` is deleted (Epic #779 M3, #787/#789) — the entry freezes when the page naturally leaves the resident union, not at a forced prompt-end closure. The `at_prompt_end` freeze and the natural-leave freeze describe the same lowest-boundary rule; read "prompt-end boundary" as "the boundary the walk reached", not as a close-time action. See [errors/2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md](../errors/2026-09-21-optimizing-at-the-wrong-layer-close-scheduling.md).
- Any snapshot strictly between can only land an intermediate grow-entry
  length. The back-scan in `publish_dropped` already skips a boundary whose
  pages have not all dropped, so dropping the middle snapshot makes the entry
  freeze at the lowest boundary instead — the follower recomputes that much
  prefix from a correct recurrent state. Tokens do not change; only how much
  prefix is reused.

Two CPU gates in `tests/test_sparse_engine.py`:

1. `..._retains_at_most_two_boundary_snapshots_per_request` — a 64-boundary
   prefill leaves ≤2 snapshots and ≤2× one snapshot's bytes. Red on main
   (entries grew with chunks), green after.
2. `..._long_prefill_snapshot_cap_keeps_a_follower_hit_exact` — a 32-page
   prefill with a one-page chunk notes a boundary per page; the prompt-end
   entry still publishes, a follower adopts the full prefix and decodes
   token-identically to a prefix-miss sparse engine.

## Expected RSS after the fix

Snapshot host bytes are bounded by two per live request regardless of context:

| ctx | snapshot bytes after | other host (bounded K/V cold, weights, staging) |
|---:|---:|---:|
| 64k | 2 × 149.6 MiB ≈ **0.29 GiB** | unchanged; measured total was 27.9 GiB with 18.7 GiB snapshots |
| 128k | 2 × 149.6 MiB ≈ **0.29 GiB** | same shape; expected peak ≈ 27.9 − 18.4 ≈ **9–10 GiB** |

The 64k/128k peaks should therefore be roughly equal and near the bounded
K/V+weights floor, not linear in context. The precise floor is device-bound
(pinned staging and the CUDA allocator's reserve are not visible from the CPU
cell), so these are predictions to check, not measurements.

## Rule

A host container the spill tier does not account for defeats the tier: a byte
budget on K/V pages is silent about a second per-request dict holding 150 MiB
per prefill chunk. The tell was `cold_ssd=0` at an RSS-pinned kill — the spill
path was healthy, the bytes were in a container it never walked. Count every
anonymous host copy against the budget, and make the count a test: per-chunk
growth on main was the failure, a constant bound is the fix.

## Pending device confirmation

```bash
scripts/v100.sh run v100_256k_ssd 'bash scripts/_v100_256k_ssd.sh'
```

Read the peak RSS at the 64k and 256k stages: it should stay near the
~9–10 GiB floor at both, and 256k should complete prefill instead of SIGKILL.
`scripts/trace_256k_spill_rss.py` additionally logs `cold_host_gib`, which this
fix does not change; the new quantity to watch is process RSS minus that.

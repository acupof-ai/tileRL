# V100 sparse 256k is SIGKILL in prefill under a 10 GiB host tier; 64k serves — 2026-09-13

> Status: **closed (code) 2026-09-14 — mechanism found and fixed on CPU;
> device RSS confirmation pending-remote.** The kill was not K/V pages
> out-running demotion: the growing bytes were the per-chunk host GDN snapshots
> in `SparsePrefixCache._snap`, held outside `HostKvPages`' budget. See
> [wins/2026-09-14-bound-sparse-prefill-gdn-snapshots.md](../wins/2026-09-14-bound-sparse-prefill-gdn-snapshots.md).
> The original observation and the (wrong) hypothesis are kept below.
> Owner: 5f. Measured by cc on the V100 box (`n37-002-027`, V100-SXM2-32GB).

## Context

Does the V100 (sm70, f16 cold pool, `--cold-format f16`) serve sparse 256k
with the host cold tier spilling to the mmap SSD? One process per context,
`sparse_k=128`, `--slots 1`, pinned host budget 10 GiB, spill file on
`/data00`. A single prompt is prefilled to N tokens then decoded 16 steps;
the same process runs 64k then 256k.

Pod launcher first line:

```
pod_run: tree /work/tilerl-s-tilerl-s-cc567 sha 9efd3833… (the probe
synced pre-fix; routing identical to the merged #567 head)
V100CAP start max_ctx=262144 host_budget=10.0 GiB ssd=/data00/sparse_cold_256k.bin
```

## Measured (2026-09-13, V100, k=128, B=1)

| ctx | result | prefill s | decode tok/s | peak RSS GiB / 31 | peak private cold SSD GiB |
|---:|---|---:|---:|---:|---:|
| 65536 | **serves** | 343.2 | 3.44 (med tick 290.6 ms) | **27.92** | **0.000** |
| 262144 | **SIGKILL during prefill** | — | — | — | **0.000** (no spill file created) |

At the 256k death `/data00/sparse_cold_256k.bin` did not exist
(`cold_ssd_gib=0.000`), and the probe process was reaped (pid gone, GPU
0 MiB). The 64k stage already peaked at 27.9 GiB RSS on a 31 GiB box with
zero pages spilled to SSD.

## Root cause (observed only — mechanism not proven)

A 256k request needs roughly 16 GiB of f16 cold KV for the own span at
full context; the 64k run peaks at 27.9 GiB RSS before any demotion. The
256k prefill was OOM-killed with cold_ssd still 0.

**Hypothesis, not yet measured:** the sparse own-span pages of a single
growing prompt are pinned into the 10 GiB host tier faster than prefill
demotes them, so the host working set exceeds the 31 GiB box before the
SSD spill path is ever reached. This needs a cold-pool byte reading at the
kill to confirm; RSS + cold_ssd=0 alone do not establish it. Alternatives
not distinguished: transient prefill staging blowup, or a leak.

## Fix options (none landed)

- larger pinned host budget / a larger-RAM box;
- demote (or spill) own-span pages eagerly during prefill instead of
  pinning the full growing span;
- cap the supported V100 sparse context at 64k for this budget/box.

## Rule

A cold-tier "spill to SSD" capacity claim requires a spill byte reading at
the failure: with RSS at the ceiling and cold_ssd=0 the SSD path was never
exercised, so "256k supported via spill" is unsupported. Do not promote a
zero-spill prefill OOM into the spill tier's verdict.

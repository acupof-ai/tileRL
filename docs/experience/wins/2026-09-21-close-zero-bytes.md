# close 零字节：请求结束不再发布 KV — sm90/H20, 2026-09-21

> Status: pending-remote

## Context

Publish-once refactor M3 (#782): a page reaches the shared prefix only when it
leaves the resident union (`offer_drop`). `_release` used to force the
prompt-end frontier closure at request finish, snapshotting every page the hot
pool never dropped from its live device frame. Under training's disjoint-span
workload those pages had zero followers and each 32k close paid ~1.8 s
ssd_mmap + ~1.8 s host→device transfer for them. The metric that matters:
close-tick `ssd_mmap` and the five `pub_*` segments at request end.

## What Worked

`engine._release` no longer calls `SparsePrefixCache.close_request` (deleted):
the forced frontier closure, its live-frame transfer loop and the
freeze/share_ref bump are gone. Prompts that stay resident for the whole run
are no longer shared; followers adopt only pages that left the pool and
published. Cancel and normal finish are equivalent for publishing. The
natural-leave freeze (`at_first` / `at_prompt_end`) is unchanged.

Expected measurement (same disjoint-span 32k close window as
`bg-publish-device-2026-09-20`): close-tick `ssd_mmap` 0 and the close
`release_close_request` bracket carries no pub_* bytes; identical steady-state
decode ms/tok; follower-hit rate on repeated shared prompts unchanged (those
pages drop naturally in decode under the k+window union).

## Rule

A content-addressed cache publishes at the moment content leaves the owning
pool, never at a request lifecycle event; forcing publication at close bills
every publisher for followers that do not exist.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-21 | a4ce0c1a | pending pod | sm90 | Qwen3.8-27B NVFP4 | | | |

Raw artifacts: pending remote run via `scripts/run_close_window_v100.sh`
(arms table cleanup is #784).

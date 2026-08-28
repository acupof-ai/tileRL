# Prefill +9%: the tick's KV descriptors crossed to the device 971 times, synchronously

> Status: Shipped

## Context

Prefill was 1836 tok/s end to end against 2058 "GPU-bound" from the same
profile, and 4908 for sglang. Every analysis of the gap had been driven off the
per-kernel GPU table, where the three big kernels account for 86.5% and nothing
looks wrong. **A GPU-busy total cannot see host stall**, which is where the
first 11% was.

The profile's own memcpy rows said it: `Memcpy HtoD (Pageable -> Device)`,
**971 times** in one 64-layer prefill. An unpinned H2D copy is synchronous — it
blocks until the copy engine drains.

## What Worked

`Engine._make_kv` built `block_table` / `seq_len` / `state_slot` /
`seq_q_lens` as host tensors once per tick, and then every kernel migrated them
itself at the tilelang boundary (`Backend._dev`) — roughly four copies per
layer, per forward. Allocate them pinned and move them once, in `_make_kv`.
`_dev` is a no-op when device and dtype already match, so nothing downstream
changes.

Measured (H20, GPU 7, `scripts/bench_harness.py --suite prefill`):

| row | before | after | |
|---|---:|---:|---:|
| prefill/len512 | 1836.3 | 2013.4 | **1.096x** |
| prefill/len2048 | 1827.4 | 1977.3 | **1.082x** |
| prefill/len8192 | 1777.5 | 1948.1 | **1.096x** |

Pageable H2D copies: **971 -> 5**. GPU-busy is unchanged (971.6 -> 970.7 ms),
which is the point: the win is entirely host-side.

## Rule

When end-to-end throughput trails the profile's own GPU-bound figure, the
difference is host time and the per-kernel table cannot show it. Read the
memcpy rows — a four-digit count of pageable copies is a four-digit count of
synchronous host stalls, and it will sit at 0.1% of GPU time while costing 10%
of the wall clock.

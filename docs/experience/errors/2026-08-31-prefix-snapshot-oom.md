# Prefix-snapshot OOM: an entry cap bounding 150 MiB tensors — sm70, 2026-08-31

> Status: Fixed

## Context

B=1 decode at 1K context returned HTTP 500. The daemon thread was dead:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 144.00 MiB.
GPU 0 has a total capacity of 31.74 GiB of which 46.38 MiB is free.
  engine.py:1165 _commit -> _publish_prefix
  engine.py:1188 self._states.states[req.state_slot].clone()
```

Short-context B=1 (31 tokens) had measured fine at 20.3 tok/s. The 4K
long-context run had also passed — it generated only 33 tokens.

## Root Cause

`_prefix_state` holds one GDN state snapshot per resident store entry, and the
27B snapshot is large:

```
48 GDN layers x 48 heads x 128 x 128 x 4B (f32 pool on CUDA) = 144.0 MiB
+ conv window: 48 x 3 x 3072 x 4B                           =   5.6 MiB
                                                              -----------
                                                               149.6 MiB
```

144.00 MiB matches the failed allocation byte-for-byte.

`_commit` publishes one every `BLOCK_TOKENS` (16) decode tokens. The dict is
bounded only indirectly: `PrefixStore.on_evict` pops the snapshot when the
store drops an entry, and the store evicts at `capacity`. But **`capacity`
counts ENTRIES, not bytes** — the 4096 default is 576 GiB of snapshots, so
eviction never fires before HBM is exhausted.

288 generated tokens at 1K ctx = 18 publishes = 2.6 GiB on a GPU already at
29.7/31.7 GiB with weights + pools. Hence OOM.

The two passing cases hid it for the same reason: both published <= 2
snapshots. Snapshot count scales with *generated* tokens, not context, so a
short-output long-context bench cannot surface it.

## Fix

Derive the store's entry cap from a byte budget in `build_engine`, on CUDA
only (`engine.py`, 8 lines):

```python
kw = {}
if backend.device.type == "cuda":
    snap = state_pool.states[0].nbytes
    if state_pool.conv_windows is not None:
        snap += state_pool.conv_windows[0, :, 0].nbytes
    kw["capacity"] = max(1, int(torch.cuda.mem_get_info()[0] // 4) // max(snap, 1))
```

`mem_get_info` is read after weights and both pools are allocated, so the
budget is real free HBM, not nominal capacity. On the V100 with the 27B it
resolves to **3 resident snapshots** — the honest number: a 149.6 MiB snapshot
against ~2 GiB of headroom leaves no room for a deep resident prefix cache.
When `kv_tier_path` is set, `on_demote` spills evicted snapshots to SSD, so a
small cap costs hit rate, not correctness.

CPU keeps the 4096 default (bf16 pool, host RAM) — `kw` stays empty, so no
test churn: 41 passed, 1 skipped.

## Results

B=1 decode after the fix, measured as a 256-token slope (32 vs 288 max_tokens)
so the prefill term cancels — a small delta is swamped by prefill variance:

| prompt_tok | decode tok/s | ms/tok | 32tok wall | 288tok wall |
|---:|---:|---:|---:|---:|
| 31 | 20.3 | 49 | 3.18s | 15.78s |
| 1052 | 8.7 | 114 | 66.86s | 96.12s |
| 2072 | 5.3 | 190 | 205.16s | 253.75s |

20 snapshots published across the 1K run, zero OOM, GPU steady at 29.7 GiB.

## Rule

A capacity that counts entries is not a bound when the entries own
heap-allocated tensors. Any cache whose value is a GPU tensor gets its cap
derived from `mem_get_info` at the allocation point, never a picked constant.

Second rule, for benching: snapshot/publish bugs scale with **generated**
tokens, not context length. A long-context bench with a short output proves
nothing about them — the 4K/33-token run passed while 1K/288 died.

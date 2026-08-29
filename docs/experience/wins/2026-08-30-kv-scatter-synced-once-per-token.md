# The KV scatter fallback synced once per token, per layer — 2026-08-30

**Date:** 2026-08-30 · **Scope:** `kv` · **Status:** CPU-verified, GPU `pending-remote`

## Context

The decode tick's host syncs were audited earlier today and cut from 7 to 4
([2026-08-30-sampler-host-syncs.md](2026-08-30-sampler-host-syncs.md)). Prefill
was never counted — which matters, because the synthetic-data workload this
engine is being built for is prefill-heavy.

## What worked

Same method: `aten._local_scalar_dense` in one tick, attributed to its frame.
One 32-token prefill chunk, tiny model, CPU:

```
host syncs in one prefill tick: 35
  32  kv_cache.py:227 write_tokens     <- ONE PER TOKEN
   1  kv_cache.py:223 write_tokens
   1  kv_cache.py:224 write_tokens
   1  reference.py:738 <listcomp>
```

`PagedKvPool.write_tokens` is called once per full-attn layer and looped per
token:

```python
for ti in range(sq):
    pos = base + ti
    blk = int(kv.block_table[bi, pos // BLOCK_TOKENS])   # device -> host
    self.k_pool[plane, blk, :, pos % BLOCK_TOKENS, :] = k[bi, ti]
```

So the cost is `b * seq_q * num_full_attn_layers`. **A 512-token prefill chunk
of the 27B is 8192 host syncs a tick** — and `r.blocks` is a Python list the
engine already holds, shipped to the device by `_make_kv` and read back one
element at a time. Third instance of that shape today, after the sampler and
Adafactor.

This is the fallback for arches without the `write_tokens` scatter kernel: the
CPU target (so CI and every local test) and any cell that has not registered it.

**Fix:** index instead of loop. `pos` as an arange, `blk`/`off` as tensors, one
advanced-index assignment per row — no per-token scalar at all.

## Numbers

| host syncs (`_local_scalar_dense`) per tick | before | after |
|---|---:|---:|
| prefill, 32-token chunk, 1 full-attn layer | 35 | **1** |
| prefill, 512-token chunk, 27B's 16 layers (derived) | 8194 | **2** |
| decode | 4 | **1** |

The one that remains in both is the eager `gdn_forward` fallback, which a cell
with `gdn_decode_fused` does not run.

**The scalar count is half the picture.** `aten._local_scalar_dense` sees
`int()` / `.item()`, not `t.tolist()` — and the two `tolist()` calls that
replaced the per-row `int(seq_q_lens[bi])` / `int(seq_len[bi])` are batch-wide
transfers that stall just the same. `scripts/probe_syncs.py` (written for this)
counts both, by wrapping the Tensor methods as well as watching the dispatch:

| tick | scalar | bulk |
|---|---:|---|
| decode | 1 (eager `gdn_forward`) | 3 — `write_tokens` x2, the sampled tokens |
| prefill | 1 | 2 — `write_tokens` |
| train step | 1 (the loss finite-check) | 0 |

So `write_tokens` still transfers **twice per full-attn layer** — 32 a tick on
the 27B, against 8194 before, and none at all on a cell with the scatter
kernel. Reporting "1 sync" without this table would have read as clean.

Tick time: **pending-remote**. A sync costs nothing measurable on CPU.

The cross-device case is why the arange starts on the block table's device and
moves after: the metal parity path holds a CPU table against an mps pool, and
building the index on the pool's device breaks it (caught by
`test_cpu_metal_decode_parity`, which went red on the first version).

## Rule

Audit the prefill tick separately from the decode tick. They take different
branches, the per-token cost only shows up in the one with tokens, and the
decode audit that ran first reported 4 syncs while prefill was running 8192.

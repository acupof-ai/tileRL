# Cost model

One primitive prices every byte and every kernel. Occupancy on the card, in host
RAM and on the SSD is derived from it, and the allocator consumes that derivation
instead of recomputing it. The invariant is

```
peak = sum(static rows, derived) + transient
```

with `derived == measured` on every static row; `transient` (attention partials,
kernel workspace, graph pools, allocator slack) is measured as the remainder and
reported as its own row, never folded into a tolerance.

## Format

`precision.Format` is the storage format of a tensor: bits per element plus zero
or more scale planes. A scale plane is `(group, dtype)`: `group` is an int for
one scale per that many elements along the last axis, or a tuple over the
trailing axes (`None` = that whole axis); a bare `None` is one per tensor.
Count = leading dims × Π ceil(trailing/group), so per-16 is `(16,)`, per-row
`((None,),)`, and a 2-D `[N/128,K/128]` grid is `(128,128)`.

```
bf16      = Format(bits=16)
f32       = Format(bits=32)
fp8_kv    = Format(bits=8,  scales=((head_dim, f32),))         # one f32 per plane x head x token
nvfp4     = Format(bits=4,  scales=((16, e4m3), (None, f32)))  # disk: ModelOpt packing
nvfp4_dev = Format(bits=4,  scales=((16, f32), ((None,), f32)))# device: f32 scales, f32/row
fp8_block_dev = Format(bits=8, scales=(((128,128), f32),))               # fp8 block grid, no row scale
fp8_dev   = Format(bits=8,  scales=(((128,128), f32), ((None,), f32)))  # fp8 grid + f32/row
nbytes(fmt, shape) = numel * bits // 8 + sum(scale_count * itemsize for each plane)
```

`nbytes` is the only byte arithmetic in the tree. `kv_cache.bytes_per_token`,
the `num_blocks` fit in `build_engine`, the draft pool charge and the ISO frame
budget call it; none of them carries an `element_size()` product of its own.
The device faces differ from disk: `renorm_fp4_scale` widens the block scale to
f32 and splits the global into a per-row epilogue, and the fp8 block grid is
`[N/128,K/128]`. The **weights row comes from the checkpoint index, not config**
— a 27B checkpoint mixes nvfp4 and fp8 linears (264/233), and which a key is
depends on its tensor names; `precision.checkpoint_weight_specs(dir)` classifies
the safetensors headers (shapes only, no weight bytes).

Checks: on the tiny model the derived pool bytes equal the storage bytes of
`k_pool/v_pool/k_scale/v_scale` to the byte, under bf16 and under fp8; the
device-face Formats equal the served tensor storage (`pack_fp4`+renorm for
nvfp4, the fp8 GEMV operands). At the 27B's 16 planes x 4 heads x 256 the
formula gives 64 KiB per token on bf16 and 32 KiB + 512 B on fp8. The 27B
resident weight total is 24.44 GB via `checkpoint_weight_specs` (pending-remote,
`TILERL_27B_CKPT`); an all-fp4 derivation from config gives ~20.3 GB and is
known-wrong because it cannot see the fp8 population.

## Plan

`engine.plan(cfg, device_free, **engine_kw) -> list[Row]` is a pure function
that lays the memory out before anything is allocated:

```
Row(tier, owner, fmt, shape, count)      # bytes = count * nbytes(fmt, shape)
tier  in {device, host, ssd}
owner in {weights, kv_pool, draft_pool, state_slots, prefix_entries, graph_pad, staging}
```

The budget rules that exist today (`free * 2/3` for the pool, `free / 4` for
`state_bytes`, `dram_bytes`, `BLOCK_TOKENS`) live in `plan`, each as the row it
produces, so no constant is unnamed. `build_engine` allocates from the plan's
rows: `num_blocks` is `plan`'s answer, not a second computation of it. Paged
attention is therefore explicit: the pool is `num_blocks x per_block` with the
fp8 scale plane inside `per_block` (no separate scale row), and a request's
occupancy is `blocks_used x per_block`. A prefix entry on the device is pool
blocks and a state slot already counted; it becomes a row only when demoted to
`host` or `ssd`, where the copy is new bytes. The trainer's tape is outside
`plan` until training shares it.

`stats()["memory"]` returns the same rows with a measured column: on cuda the
`memory_allocated` difference around each owner's allocation, on cpu the storage
sum, and `transient = max_memory_allocated - sum(static rows)`. A dry-run mode of
`tilerl serve` prints the plan as JSON without a card. The gate is
`derived == measured` for `weights` and `kv_pool` on the tiny model; a nonzero
delta on a static row on the 27B is an error entry, not a tolerance. Nothing here
runs in the tick: `plan` is arithmetic at build time, `stats` reads counters.

## Kernel cost

Each launched kernel declares two pure functions next to its registry entry:

```
bytes_moved(shape) -> int      # in terms of nbytes(fmt, ...) of its operands
flops(shape)       -> int
bound(shape) = max(bytes_moved / bandwidth, flops / peak)
```

`bandwidth` and `peak` are one calibration row per card in the bench ledger (a
copy kernel and a large GEMM), never a datasheet number. A kernels mode of `tilerl bench`
prints `kernel / shape / bytes / flops / bound / ms / % of bound` for the kernels
one 27B tick launches, and the tick's cost is their sum. Without a card the
table prints from the declarations alone and the measured columns read
`pending-remote`. The gate is that the attention decode kernel's `bytes_moved`
equals the KV bytes the pool hands it, derived through the same `nbytes`.

## What this replaces

The three hand-rolled byte products (`kv_cache.py`, `engine.py` twice), the
bandwidth arithmetic inside the `bench_*/probe_*/profile_*` scripts, and the
percent-of-HBM claims that live only in `docs/experience/`. A number in a wins
or errors entry that the model cannot reproduce is corrected or marked stale.

## Ownership

| Unit | Owner | Depends on |
|------|-------|------------|
| `Format`, `nbytes`, three call sites, byte-equality gate | cc | — |
| `plan`, `build_engine` consumes it, serve dry-run, `stats()["memory"]` | 52 | Format |
| `bytes_moved` / `flops` per kernel, bench kernels mode, calibration row | 5f | Format |
| Recompute the recorded numbers, delete superseded probes | fourth executor | all three |

Each unit is one PR, reviewed by a non-author on goal fit, entropy (fewest
lines, no field without a consumer, reuse of `stats()` and the ledger schema) and
the 27B path. GPU columns stay `pending-remote` until the cards return.

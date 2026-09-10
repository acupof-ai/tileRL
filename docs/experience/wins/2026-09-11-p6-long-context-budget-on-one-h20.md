# P6 long-context on one H20: 256k single-stream fits in bf16; 8×128k needs fp8 — derived, 2026-09-11

> Status: **derived**, header-only arithmetic on pod CPU
> (`serve --dry-run --checkpoint`, no card). The free-byte anchor is the one
> measured H20 figure; every other column is the ledger's formula. Measured
> occupancy at these shapes stays pending-remote.

## Inputs

- Card: H20, **95.2 GiB total**. Free after CUDA context + loaded tilelang
  modules, before weights: **94.91 GiB** measured
  (`errors/2026-09-08-three-ooms-and-the-traceback-was-not-the-consumer.md`:
  "after _build_model(keep_master=True) … free 94.91", probe_rl_budget.py). The
  dry-run fits from this same post-context number, so no card is needed to
  reproduce it.
- Weights (served): **24,436,981,888 B = 22.759 GiB**, the exact
  `checkpoint_weight_faces` total (== live load_hf; #465/#466).
- KV block: **1,048,576 B (1.000 MiB) bf16**, 16 tokens/block, 16 full-attn
  planes × 4 KV heads × 256 head_dim (`per_kv_block_bytes`). fp8 KV block is
  0.5078 MiB (half + the per-head_dim scale planes).
- State slots: B+1 (the decode graph's replay row on CUDA); 1.364 GiB at 9
  (B=8), 0.303 GiB at 2 (B=1) (`_state_bytes`, reproduced by the dry-runs
  below).
- Pool rule: the fitter spends `free*2/3` of what remains after weights +
  state; the other quarter (`free/4`) is the resident-snapshot (prefix-entry)
  budget. Per GDN snapshot 156.9 MiB measured
  (`wins/2026-09-07-the-dram-tier-is-357x…`).

## The table (derived)

Fitted pool after fixed rows, then whether the named context fits for B rows
simultaneously:

| B | KV | fitted pool | tokens | uniform ctx/row | 32k | 128k | 256k |
|---:|:--|---:|---:|---:|:--|:--|:--|
| 1 | bf16 | 49,048 blk / 47.90 GiB | 784,768 | 784,768 | FIT | FIT | **FIT** |
| 8 | bf16 | 48,323 blk / 47.19 GiB | 773,168 | 96,646 | FIT | MISS (need +16.8 GiB) | MISS (+80.8) |
| 1 | fp8 | 96,587 blk / 47.90 GiB | 1,545,392 | 1.55 M | FIT | FIT | FIT |
| 8 | fp8 | 95,160 blk / 47.19 GiB | 1,522,560 | 190,320 | FIT | **FIT** (14.7 GiB spare) | MISS (need +17.8 GiB) |

Requirement for B equal contexts is `B × ceil(ctx/16)` blocks. At B=8 the
shared pool (48k bf16 blocks) holds ~96.6k tokens per row: enough for 8×32k,
short of 8×128k (needs 65,536 blocks) and 8×256k (131,072).

### Rows per fit (`--device-free 94.91GiB`, B=8, bf16)

| row | bytes | note |
|---|---:|---|
| weights | 22.759 GiB | checkpoint served faces |
| state_slots | 1.364 GiB | 9 slots (8 + graph pad) |
| kv_pool | 47.19 GiB | 48,323 blocks (fitted free×2/3) |
| kv_pool_budget | 47.19 GiB | rule free×2/3 |
| prefix_entries_budget | 17.7 GiB | rule free/4 |

## Prefix-cache capacity (derived)

- **On device**: the `free/4` budget after fixed rows is ~17.7 GiB → **115
  resident GDN snapshots** at 156.9 MiB each (the 09-07 measured count is 116
  from a marginally larger free; same answer). These are the demotion slots
  that keep a returning session's state hot; KV itself stays in the shared
  pool.
- **DRAM tier**: this pod reports 1,928 GiB total / **1,866 GiB available**
  (`free -g`, 2026-09-11), 1,478 GiB currently free (1,911,035 / 1,513,927 MiB
  by `free -m`). At 156.9 MiB that is **9,649 (spare-free) to 12,180 (all
  available)** snapshots — two orders of magnitude beyond the HBM tier; prefix
  state is not the binding resource.

## P6 verdicts (derived; pending-remote for measured)

- **256k single card, B=1: FITS** in bf16 (needs 16,384 blocks / 16.0 GiB;
  fitted pool 47.9 GiB, 31.9 GiB headroom). No fp8, no offload.
- **128k prefix cache: FITS** in every column asked: single-stream trivially;
  **8×128k fits only with fp8 KV** (65,536 blocks ≤ 95,160, 14.7 GiB spare) —
  bf16 stops at ~96k/row. The HBM snapshot budget alone holds 115 sessions'
  demoted state; DRAM holds ~9.6–12.2k.
- 8×256k does not fit on one card under either KV dtype (needs 128 GiB of KV;
  even fp8 is 17.8 GiB over the fitted pool and would require giving the pool
  more than the free×2/3 rule currently grants).

## Caveats on the derived number

- The 2/3 fitted pool is the *serving* rule that reserves the last third for
  attention partials, graph scratch and allocator slack; a 256k prefill's
  transient activation is not in this static fit and is the thing most likely
  to bind first (separate from the capacity verdict). The measured transient
  on the 09-08 run grew superlinearly with B and was unlocated.
- This is the shared-pool, no-draft shape (`spec_steps=0`). A draft depth>1
  adds `draft_per_block_bytes` to the fit denominator.
- Measured columns (an actual 256k/128k serve on the H20) remain
  pending-remote per the GPU recall.

## Rule

Single-card long-context is a KV-pool budgeting question the ledger answers
before running: state and weights are fixed, the 2/3 pool rule decides
capacity, and the B×ctx product is what must fit. fp8 KV is the one lever that
moves the 8×128 line; it is not needed for 256k at B=1.

# GDN prefill chunk kernel: 11.01 -> 0.22 ms/tok (49.8x) — cuda/sm90, 2026-08-24

> Status: Shipped

## Context

Prefill was the single blocker to the 3800 tok/s target: the 512-token slice
tick was 5766 ms, and `linear_attn_chunk` was 5657 ms of it (98.1%) —
`reference.gdn_forward` runs a Python loop over 48 value heads with ~8
einsums each, ~150k tiny kernel launches per prefill. The decode path was
already fused (`make_gdn_decode_fused`, T=1); prefill (T>1) still used the
torch-eager reference.

## What Worked

- **`make_gdn_chunk_fused`** (`kernels_mma.py`): the T-loop generalization of
  the decode kernel. One block per (value head, batch); thread `tv` owns state
  column `S[:, tv]`; a serial scan over T tokens carries the state in HBM
  (decay-first recurrence — decay the state, then apply the delta — matching
  `reference.gdn_forward` exactly). Same fused ops as decode: conv1d + SiLU +
  q/k L2-norm + decay-first delta recurrence + gated RMSNorm + z-gate. The
  whole layer core is one launch for all 48 heads.
- **Serial-within-block, not chunkwise-WY.** The tilelang branch's prefill
  path (`qwen36_prefill_wy.py` + `qwen36_prefill_scan_o.py`) is chunkwise-WY;
  tileRL's decay-first recurrence is serial-within-block instead — within a
  chunk scan serially over T steps, across chunks carry the state (input
  `State` / output `NewState` are the carry). Not fla's chunk delta rule
  (that freezes chunk-start state — incompatible with decay-first).
- **Conv history read per tap from HBM**, like the decode kernel. Two
  alternatives failed: a per-thread sliding window in shared memory races
  (each thread owns a different channel; shared is block-wide), and fragments
  forbid the `rq[i] = rq[i+1]` shift (tilelang's uniform-index constraint —
  a fragment must be indexed by the same expression everywhere).
- **Registered in the sm90 cell** (`backend.py`): `linear_attn_chunk`
  dispatches T>1 to the chunk kernel; T=1 keeps the decode kernel; other
  arches keep the torch-eager reference.
- **Parity gates** (`tests/test_ops_parity.py`): `test_gdn_chunk_fused_parity`
  (T=6, chunk kernel vs `reference.gdn_forward`, allclose rtol=1e-2) and
  `test_gdn_chunk_matches_decode` (T=1, chunk kernel vs decode kernel — the
  chunk kernel is its T-loop generalization). CUDA: 25/25 green; CPU suite
  67 passed, 2 skipped.
- **Measured delta** (single-process A/B, same GPU, JIT-free,
  `scripts/bench_gdn_prefill.py`, 2-layer NVFP4 slice, H20):

| arm | ms/tok | tick ms |
|---|---:|---:|
| torch-eager GDN reference | 11.0137 | 5639.0 |
| fused GDN chunk kernel | 0.2212 | 113.3 |

49.8x on the prefill tick; 0.2212 ms/tok is 16% under the 0.263 ms/tok
(3800 tok/s) slice target. The reference arm (11.01) reproduces the
documented 11.26 ms/tok, validating the harness. The GDN op is now ~4 ms of
the 113 ms tick — the fp4 gemms at M=512 are the remaining prefill cost.

## Rule

For a decay-first recurrent layer, the prefill kernel is a serial-within-
block chunk scan — one block per (head, batch), serial over T, state carried
in HBM across chunks; it is the T-loop generalization of the fused decode
kernel, not a chunkwise-WY rewrite. And a per-thread sliding window cannot
live in tilelang shared memory (block-wide race) or fragments (uniform-index
constraint) — read the conv history per tap from HBM, same as decode.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-24 | 53c1398 | H20 | cuda/sm90 | 27B slice (2 GDN layers) | 11.26 (512-tok) | 31.09 | 89 prefill |
| 2026-08-24 | (this) | H20 | cuda/sm90 | 27B slice (2 GDN layers) | 0.2212 (512-tok) | 31.09 | 4520 prefill |

Prefill is the single-process A/B from `scripts/bench_gdn_prefill.py`
(reference arm 11.0137 ms/tok in the same run); decode is unchanged (the
chunk kernel is T>1 only). The 3800 tok/s slice target is met.

Raw artifacts: pod `/work/tilerl`, `scripts/bench_gdn_prefill.py` stdout
(H20, GPU 1, JIT-free).

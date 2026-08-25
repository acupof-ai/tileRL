# GDN chunk kernel: local state column + bf16 IO, 2.32 -> 1.82 ms (21.6%) — cuda/sm90, 2026-08-25

> Status: Shipped

## Context

The GDN chunk prefill kernel (`make_gdn_chunk_fused`) was 27.6% of the
prefill-512 tick (1174 tok/s, target 3800). The WY chunkwise transform was
already rejected (2.6x slower — serial is compute-bound, state fits L2; see
`errors/2026-08-25-gdn-prefill-wy-rejected.md`). The lever is the serial
scan's peak efficiency, not a different algorithm.

The kernel streams the 128-float state column through global memory 4x per
token (pass-1 load+store, pass-2 load+store — strided, 512B per thread).
Sweep 1 proved the kernel is load-latency-bound, not FMA-chain-bound:
parallel norm/rms reduces gave 0% (the dependency chains are not the
bottleneck; the strided state loads are).

## What Worked

- **Local state column** (`T.alloc_local((K,), "float32")` per thread):
  carries the 128-float state column across all T tokens in registers/L1.
  Cuts global state traffic from 512 strided accesses/token (2 loads + 2
  stores) to 128 stores/token (pass-2 final state only).
- **4 accumulators per pass** (`T.serial(K//4)` + `T.unroll(4)`): breaks the
  128-deep FMA chain into 4x32-deep AND issues 4 state loads per iteration
  (ILP hides L1/register latency). 8 accumulators tested worse (register
  pressure). The fused-dot reassociation (pass-1 accumulates both dots,
  pass-2 chain-free) tested worse — extra accumulators/allreduce overhead
  cancels chain-breaking when the kernel is load-bound.
- **bf16 IO** (Q/Key/Val/Z/Window/NewWindow are bf16; state/out/weights stay
  f32): halves the conv/projection load traffic. Stacks on local state for
  3.1% on top of the 19.1% from local+4acc alone. Parity OK (rel-err 2.7e-3
  < 1e-2).
- **sm90-only**: `gdn_chunk_fused` lives only in the sm90 cell; CPU/metal
  use `linear_attn_chunk` (unaffected). Backend casts q/k/v/z/window to
  bf16 at the boundary, casts NewWindow back to f32 for the caller.

## Rule

For a serial-scan recurrent kernel that is load-latency-bound (not
chain-bound), carry the per-thread state column in a `T.alloc_local` array
across all T steps — not streamed through global. Add multi-accumulator
unrolling to issue multiple loads per iteration (ILP hides L1 latency).
bf16 IO on the input projections stacks when the conv reads are a
significant fraction of the remaining load traffic.

## Results

Kernel-level sweep (same run, quiet GPU, `scripts/_sweep_gdn_prefill2.py`,
B=1 T=512 QD=2048 NVH=48 K=V=128, H20):

| variant | ms | rel-err | vs baseline |
|---|---:|---:|---:|
| baseline (global state, f32 IO) | 2.3234 | 3.1e-7 | — |
| local + 4acc (f32 IO) | 1.8801 | 1.8e-7 | -19.1% |
| local + 4acc + bf16 IO | 1.8222 | 2.7e-3 | -21.6% |

Slice4 profile (quiet GPU, `scripts/profile_slice.py`, 3 GDN + 1 FA layers,
prefill-len 512):

| metric | before (est.) | after (measured) |
|---|---:|---:|
| GDN chunk op (ms) | 8.08 | 6.34 |
| slice prefill (tok/s) | ~18300 | 18759 |
| full-model prefill (tok/s) | ~1200 | 1224 |

Before estimated from the sweep ratio (21.6% kernel improvement applied to
the measured after); the direct before-profile ran under GPU contention
and is not comparable. The end-to-end gain is small (~2.4%) because the
GDN chunk is ~24% of the slice GPU sum — the 21.6% kernel improvement
translates to ~5% on the slice.

Parity: 4/4 GDN tests pass on CUDA (chunk vs reference, chunk vs decode,
decode vs reference, conv-window). Ruff clean.

Raw artifacts: pod `/work/tilerl`, `scripts/_sweep_gdn_prefill2.py` stdout
(H20, GPU 0, quiet window, JIT cached).

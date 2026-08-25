# Chunkwise-WY GDN prefill rejected — 2.6x slower than serial scan

> Status: Killed (correctness green, performance regression)

## Context

The serial GDN prefill chunk kernel (`make_gdn_chunk_fused`, serial-within-block
scan over T tokens, state streamed from HBM twice per token) was 27.6% of the
prefill-512 tick. The tilelang branch `feat/qwen36-gdn-megakernel` has an
unmerged chunkwise-WY prefill pair (`qwen36_prefill_wy.py` +
`qwen36_prefill_scan_o.py`) that replaces the per-token scan with a
chunk-parallel (I+L) triangular solve. Task: port it, wire it, bench it.

The port was correct (parity green vs `reference.gdn_forward`, rtol=1e-2,
T=1 cross-check vs the decode kernel) but **2.6x slower** than the serial
kernel on the H20 pod:

| kernel | GPU ms (min, T=512, slice4 shapes) |
|---|---:|
| serial chunk (existing) | 4.73 |
| WY kernel A (chunk-parallel solve) | 1.62 |
| WY kernel B (chunk-serial scan) | 10.76 |
| WY total | 12.38 |

## Root Cause

The WY formulation is fundamentally **O(T·C·K)** per head (C=64: the A
construction, u/w matmuls, QK, and output are all C²-contractions), while the
serial scan is **O(T·K)**. For C=64 that is 64x more FMAs. The WY's
chunk-parallelism (8 chunks in flight) was supposed to repay this, but:

1. **The serial kernel is compute-bound, not memory-bound.** Its state tile is
   K·V·4 = 64KB per head, 48 heads = 3MB — fits in L2. The "state streamed
   from HBM twice per token" is actually L2 traffic (~5TB/s), ~13us for the
   whole prefill. The serial dependency chain (512 tokens × ~5 dependent ops)
   is the real cost, and it is short enough.
2. **48 value heads already saturate the SMs.** The serial kernel launches 48
   blocks (one per head); the H20 has 78 SMs. The WY's 384 blocks (8 chunks
   × 48 heads) don't add parallelism where it's needed — kernel B is
   chunk-*serial* (state dependency), so it runs 48 blocks over 8 sequential
   chunks, each doing C² work.
3. **Two launches + 24MB intermediates** (u/w/gcs between the kernels) add
   overhead the single serial kernel doesn't have.

The WY wins when the serial kernel is *memory-bound* (state doesn't fit in
cache) or *parallelism-starved* (few heads). Neither holds for Qwen3.6-27B
(48 heads, 64KB state).

## What was learned (kept for future kernel work)

**Recurrence verdict.** The branch's WY *solve* (kernel A: A[i,j] =
beta_i·exp(gcs_i−gcs_j)·(k_i·k_j), M = (I+StrictLower(A))⁻¹, u = M(v·beta),
w = M(k·beta·exp(gcs))) is **exactly the decay-first delta rule** — the
chunk-start state is decayed per-token inside the solve (not frozen like fla's
chunk delta rule). Verified f32 vs the serial recurrence to 1e-17 (deltas).

But the branch's kernel B (state update + output) implements a **lagged
recurrence**, inconsistent with its own decode kernel:
- state: branch uses unweighted `h' = h·e_last + K^T d`; decay-first needs
  `h' = h·e_last + K^T (D_last ⊙ d)` with `D_last[r] = exp(gcs_last − gcs_r)`.
- output: branch uses strict-lower intra (`p<r`) + double q-scale; decay-first
  needs diagonal-inclusive (`p<=r`, post-update state) + single scale.

Both corrected in the port; verified 5e-9 (out) / 2e-8 (state) f32.

**tilelang 0.1.13 miscompiles thread-divergent branches** (`if thread_binding ==
loop_var`) inside serial loops — silently wrong, no error. Three hits:
1. `if ti == r` in the forward-substitution solve → M stayed identity.
   Fix: column-parallel solve (all threads compute their column in lockstep,
   no branch).
2. `if r % V == tv` guarding a double-nested loop (QK) → rows 1+ zeroed.
   Fix: direct row assignment (`if tv < CHUNK`), valid when V ≥ CHUNK.
3. `if tv == r % V` in the RMS reduce → wrong rms. Fix: `if tv == 0` (any
   thread can reduce the shared buffer).

The same `r % V == tv` pattern with a *single* inner loop (norm, gcs load)
compiles fine — the failure needs the nested loop.

## Rule

Don't replace a serial scan with a chunkwise WY scan when the state fits in
L2 and heads ≥ SMs/2 — the O(T·C) extra FMAs and two-kernel overhead cost
more than the state-traffic savings. WY pays off only when the serial kernel
is memory-bound (state > L2) or parallelism-starved (few heads). Measure
both; the 27.6% tick share was state-L2 traffic + serial latency, not HBM.

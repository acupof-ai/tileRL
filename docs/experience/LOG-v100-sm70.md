# V100 sm70 fp4 — working log

Running record for this worktree. Newest entry at the bottom of each day.
Numbers only; the reasoning that produced them lives in
`docs/experience/{wins,errors}/`.

## 2026-08-31

**Server prefill batching shipped** (commit 5be4042, earlier session).
B=8 through the server 2.8 → 12.9 tok/s. Long-context B=1 TTFT 61/199/687s at
1K/2K/4K — O(T²) from the GDN serial scan, inherent to the recurrent update.

**Wikitext-103 perplexity = 8.67** (`scripts/bench_c4_ppl.py`, 5 chunks, 2555
tokens, B=1 teacher-forced). C4 is not cached on the pod and both offline and
online loads failed (`ConnectionError`, then `httpx.InvalidURL: Invalid port`),
so wikitext-103-raw-v1 test is the substitute. **Open:** ckl expects ~26 —
unresolved which dataset/baseline that refers to. Asked, not yet answered.

**Prefix-snapshot OOM found and fixed** (commit 5b5fc0a).
`PrefixStore.capacity` counts entries; each resident entry owns a 149.6 MiB GDN
snapshot in HBM. 4096 default = 576 GiB, so eviction never fired — B=1 at 1K
ctx died after 18 publishes on a 144.00 MiB alloc. Cap now derived from
`mem_get_info` at build time (CUDA only) → 3 resident snapshots on the V100.
Entry: `errors/2026-08-31-prefix-snapshot-oom.md`.

**B=1 decode measured** (256-token slope, so the prefill term cancels):

| ctx | tok/s | ms/tok |
|---:|---:|---:|
| 31 | 20.3 | 49 |
| 1052 | 8.7 | 114 |
| 2072 | 5.3 | 190 |

**Short ctx is at roofline, not slow.** V100 900 GB/s vs H20 4000 GB/s = 4.44×;
measured 87.5 → 20.3 tok/s = 4.31×. 97% bandwidth efficiency. No fp4/bf16
tensor cores on sm70 and decode is memory-bound — 20.3 is what the hardware
owes.

**The ctx scaling IS broken — root cause found.**
`kernels.py:629` generic `paged_attention` (sm70's only path):
`T.Kernel(B, H)` = 24 blocks on an 80-SM card, and the dot product is
`T.serial(D)` so each block runs **one** active thread.

Two-point slope cancels the ctx-independent terms (GEMM + GDN):
(190−114) ms ÷ (203.7−103.4)M FMA = **0.76 ns/FMA** ≈ 1.16 clocks @1.53 GHz —
the theoretical speed of a single-threaded serial scalar loop. Direct proof.
The same KV is 134 MB → 0.15 ms at bandwidth. 1000× off.

sm90 never exposed it: it dispatches to `paged_attention_decode` (split-KV
flash-decoding). Same bug class as `gdn_chunk_fused` being registered only in
`_SM90_KERNELS`.

**Fix shipped: split-KV decode attention, sm70 cell only.**
`T.Kernel(KVSPLIT, H, B)` + log-domain combine. Parallelism from the grid, not
a fragment reduction, so `T.serial(D)` stays and the Metal constraint holds.
Cannot reuse the sm90 kernel (needs `T.gemm` + bf16).

| ctx | before | after | speedup |
|---:|---:|---:|---:|
| 31 | 20.3 | 25.8 | 1.27× |
| 1052 | 8.7 | 23.1 | 2.65× |
| 2072 | 5.3 | 20.5 | 3.87× |

Falloff 3.8× → 20%. Parity 1.8e-07 on sm70, 2.4e-07 on CPU; 146 tests pass.
Registered for sm70 only + dispatch arch-gated — first draft had it in
`_CPU_KERNELS`, which cpu/metal/rocm all inherit. Entry:
`wins/2026-08-31-sm70-split-kv-decode-attention.md`.

**Roofline ceiling, for the 60 tok/s question.** 27B NVFP4 weights ~14 GB per
token / 900 GB/s = **15.6 ms/tok = 64 tok/s hard ceiling**. Now at 39 ms/tok
(23.4 ms of non-weight overhead). 60 dense = 94% of roofline: not reachable.
The H20's 87.5 tok/s was against a 285 tok/s roofline — 60 was 21% there.
Dense headroom left: maybe 35-40 tok/s. **60 requires speculation** — 2.34
accepted tokens per forward at the current 39 ms.

**MTP head is in the checkpoint**: `mtp.fc.weight` + `mtp.layers.0.*` (one
full-attn layer, 2193 keys total). Jointly trained with the trunk, so no
training needed and acceptance should beat a bolted-on draft. Speculation is
also *cheaper* on V100 than on H20: decode is fully weight-bandwidth-bound
here, and `linear_fp4_gemv_sm70_m` (M=8) already reads W once for 8 rows, so
the extra draft rows are nearly free.


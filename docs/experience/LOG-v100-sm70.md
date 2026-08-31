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

**MTP head works; speculation still loses. Root cause: the draft step is
outside the captured graph.**

Checkpoint MTP head loaded via the existing `load_draft` — all 15 `mtp.*` keys
map cleanly, and they all live in one shard (`model-00018-of-00018`). Quality is
good: **62% top-1 agreement** with the trunk, median trunk-rank 0, 84% in
top-5. Accept rate in serving 97-99%, **5.33 tokens committed per forward**.

But measured 3.1 tok/s at depth 6 vs 25.8 dense. `prof_draft_step.py`:

| | ms (M=1, eager) |
|---|---:|
| trunk forward | 103.58 |
| draft step (1 layer, 456 M) | 120.91 |
| draft bandwidth floor | 0.25 (fp4) / 1.01 (bf16) |

The 1-layer head costs as much as the 64-layer trunk — both launch-bound at
M=1. The trunk hides it behind graph capture (103.58 eager → 39 captured,
2.66×); the draft loop runs outside, so it pays eager per step. Predicted 3.3
at depth 6, measured 3.1.

Capture the draft step and depth 6 projects to **62.7 tok/s**. Entry:
`errors/2026-08-31-draft-step-outside-graph.md`.

Two wrong turns worth remembering, both inferred from end-to-end throughput and
both killed by one direct measurement: fp8 quantization has no sm70 kernel
(real, 0.7 → 6.0 tok/s, still a loss) and then fp4 quantization (made it worse,
3.1 — at M=1 nothing is bandwidth-bound so the format is irrelevant).

Two capacity facts found on the way: `step_states` is sized by SLOT COUNT
(`16 slots × 7 steps × 144 MiB = 15.75 GiB` OOM'd the card; 4 slots works), and
tree verification is blocked because `kernels_gdn.py:500-520` evolves
`state_local` across the `t` loop — node t builds on t−1, not on its parent. A
linear chain tops out at `1 + p/(1−p) = 2.63` tokens = 67.5 tok/s at p=0.62, so
no tree is needed for 60.



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

**Roofline ceiling, for the 60 tok/s question.** *(Corrected 2026-09-02 — this
entry originally cited a remembered ~14 GB; see
`errors/2026-09-02-roofline-is-the-streamed-subset.md`.)* A dense token streams
**16.04 GB** — trunk layers 15.24 + lm_head 0.80. `embed_tokens` (2.54) and the
visual tower (0.92) are resident but not streamed, so the 20.35 GB checkpoint is
not the denominator. 16.04 / 900 GB/s = **17.8 ms/tok = 56.1 tok/s hard
ceiling**. Dense is now 35.3 tok/s at 4096 ctx = 63% of it. 60 dense is above
the ceiling and unreachable; **60 requires speculation** — spec peaks at 50.8
(1024 ctx) on 3.26 tok/forward.


**MTP head works; speculation still loses. First diagnosis (draft outside the
graph) was real but minor — see the corrected root cause at the end.**

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
(real, 0.7 → 6.0 tok/s, still a loss) and then fp4 quantization (appeared to
make it worse, 3.1 — **later shown to be my profiler's bug**, see below; fp4 is
in fact 24× faster per draft step).

Two capacity facts found on the way: `step_states` is sized by SLOT COUNT
(`16 slots × 7 steps × 144 MiB = 15.75 GiB` OOM'd the card; 4 slots works), and
tree verification is blocked because `kernels_gdn.py:500-520` evolves
`state_local` across the `t` loop — node t builds on t−1, not on its parent. A
linear chain tops out at `1 + p/(1−p) = 2.63` tokens = 67.5 tok/s at p=0.62, so
no tree is needed for 60.

**Speculation traced to the bottom: the GDN state gather/scatter, not anything I
guessed first.** Pure graph replay (no observer effect): W=1 39 ms, W=3 266 ms,
W=4 267 ms. **Flat in W** — so W>1 is a path switch, not a scaling term.
`backend.py:917` routes any `q.shape[1] > 1` to `gdn_chunk_fused`, and on sm70
`gdn_decode` is None (sm90-only), so `model.py:421-439` does
`state_gather` -> kernel -> `state_scatter` per layer. `reference.state_gather`
is `states[slots, layer_idx]` torch advanced indexing: 3 MiB out + 3 MiB back
x 48 layers = 288 MiB of unfusable round trip, 4.73 ms/layer. W=1 pays it too,
which is why 39 ms is already 2.5x the 15.6 ms roofline.

Fix would be extending `gdn_decode_fused` to T>1 with in-place pool state — a
kernel project. Speculation stays OFF. Entry:
`errors/2026-08-31-spec-blocked-on-gdn-state-path.md`.

Shipped on the way: split-KV attention now covers S>1 (9.5x at S=4/n=1024;
attention turned out to be 2.7% of the tick), and the draft head is fp4-quantized
on sm70 — 4.98 ms/step vs 120.91 dense (24x).

Five wrong turns, all inferred from end-to-end throughput and all refuted by a
direct measurement. Two worth remembering: an op-timing wrapper that calls
`torch.cuda.synchronize()` breaks graph capture and then measures the fallback it
caused (it told me `linear_fp4` was 69% of the tick — it was not), and my fp4
profiler deleted the dense keys including `fc`, silently reverting to dense, which
made me reject fp4 as slower when it is 24x faster.

**CORRECTION — the blocker is one GEMV kernel, not GDN.** `profile_verify_replay.py`
(already in the tree) attributes the verify replay per kernel:

| W | replay | GEMV µs/call | GEMV total |
|---:|---:|---:|---:|
| 1 | 40.9 ms | 64.5 (`..._sm70`) | 32.1 ms |
| 2 | 271.0 ms | 507.9 (`..._sm70_m`) | 252.4 ms |
| 8 | 271.6 ms | 507.5 | 252.2 ms |

GDN was 1.9-2.9 ms, attention 1.7 ms. So my gather/scatter root cause was WRONG —
the whole 230 ms step is `linear_fp4_gemv_sm70_m`, and it is flat in W (W=2 pays
the full M=8 tile).

`scripts/ab_m8_reuse.py`: the M-row kernel's tile reuse works at N=K=4864 (1.65x
for 8 rows) and fails at every 27B projection (6.2-7.5x) — where M=8 is worse
PER ROW than M=1. 4864 is exactly the shape
`wins/2026-08-30-sm70-fp16-twiddle-gemv.md` benchmarked; the kernel was never
timed at the production shapes. Fixed at 1.65x, W=8 verify would be ~73 ms and
depth 3 would give 30.7 tok/s, past dense. Entry:
`errors/2026-08-31-m8-gemv-occupancy-not-reuse.md`.

Ruled out for the 6-7x: weight re-decode (the decode is outside the `for m`
loop), occupancy (>1200 blocks everywhere), converts (1280/thread at both K),
X re-reads (0.8 ms of the ~1.0 ms gap — a contributor, not the cause). Next is
ncu, not more arithmetic.

**Workflow scoping (3 lanes) landed two corrections worth keeping.** My "an extra
verify row costs 0.021 ms" was activations only and ignores ARITHMETIC: no fp4
tensor cores, so each row redoes 37.3 GFLOP = 2.38 ms at 15.7 TFLOPS. Verify goes
compute-bound past M~6.5, so the optimum width is M=8-16 and "32 candidates for
4% more" was wrong; 60 tok/s needs 3.08 accepted tokens/forward at M=8, not 2.34.
And tree TOPOLOGY is not data-dependent — fixed at capture, the parent/mask
tensors are constants, so my graph-capture worry was unfounded. A cheaper tree
shape also exists: k independent chains sharing the committed root, laid out
contiguously, makes `Parent[t]==t-1` hold everywhere except k chain heads (~8
lines in GDN) and turns the ancestor mask into one int32 offset vector.

## 2026-09-01

**Packed-f16 X — the GEMV's per-row floor was never the hardware.**
127 us/row flat was X being re-read per block (78% of the M=8 bytes) and
converted f32->f16 inside the tile loop (32 of ~49 per-row instructions).
Pack once at the dispatch site: 34.0 us/row at M=8, 88.2 at M=1, bit-exact.

| shape | M=1 | M=2 /row | M=4 /row | M=8 /row |
|---|---:|---:|---:|---:|
| 17408x5120 | 128.1 -> 88.2 | 120.2 -> 44.9 | 126.9 -> 33.8 | 127.5 -> 34.0 |
| 12288x5120 | 92.5 -> 82.4 | 87.4 -> 33.6 | 90.8 -> 24.8 | 90.7 -> 24.2 |
| 5120x17408 | 133.9 -> 93.4 | 129.1 -> 56.7 | 133.0 -> 37.1 | 122.9 -> 37.2 |

Verify replay W=1 38.6 -> 30.9 ms; marginal verify row 63 -> 10.7 ms.
Dense B=1 server: 32.7 tok/s @31 ctx, 27.4 @1K (was 25.8 / 23.1).
Entry: `wins/2026-09-01-sm70-gemv-packed-x-f16.md`.

**Speculation blocker moved to the draft head.** `--depth 3` serves 1.3 tok/s
against a 41.6 ms W=2 replay. `prof_spec_tick.py`: `_draft_step` is 16.7 s of
21.2 s (79%), 371 ms per depth step vs 4.98 ms in isolation; `decode_graph`
636 ms/tick vs a 41.6 ms replay of the same graph. No capture-failure warning,
so the graph replays — the cost is around it. Next target.

**Speculation ACCEPTED — 52.7 tok/s at 31 ctx, 35.4 at 1K.**
Once a verify row cost 10.7 ms against a 30.9 ms token, depth 3 pays:

| ctx | dense | spec depth 3 | |
|---:|---:|---:|---:|
| 31 | 32.7 | **52.7** | 1.61x |
| 1052 | 27.4 | **35.4** | 1.29x |

100% draft acceptance in serving (292/292 on the counting task), 2.95
tok/forward. 52.7 is 82% of the 64 tok/s weight roofline.

The "1.3 tok/s" scare was a MEASUREMENT artifact: `bench_b1_decode.py` warms up
with one `--lo` call, which captures only the width-1 graph. The first spec tick
then captures three more widths inside the timed `lo` point — 2589/906/731 ms on
ticks 1-3 — and the two-point slope inverts. Per-tick timing showed the engine
was flat at ~78 ms/tick out to 317 tokens the whole time. Warm every width
before timing anything speculative.

**Realistic workloads: speculation wins on code, breaks even on chat.**
The counting task was hiding everything — it is near-zero-entropy under greedy
decode, so it accepts every draft. Against a dense baseline on the same prompts
(`scripts/bench_workloads.py`):

| workload | prompt | dense | spec d3 | |
|---|---:|---:|---:|---:|
| counting (control) | 29 | 32.7 | 52.7 | 1.61x |
| coding | 89 | 32.6 | 43.4 | 1.33x |
| dialogue | 178 | 31.9 | 32.0 | 1.00x |
| thinking | 174 | 32.0 | 30.2 | 0.94x |

**100% acceptance was an artifact.** `verify_lens` truncates the chain on the
draft's own confidence BEFORE the counter runs (engine.py:1068 -> :1074), so
accepted/drafted measures the truncation policy, not the head. Ticks where the
policy keeps nothing vanish from both counters (engine.py:795). The honest
metric is tokens per trunk decode forward; `bench_workloads.py` reports that and
no longer prints a ratio as a headline.

**Depth 3 is the ceiling, depth 4 is a cliff** — the verify width 1+depth rounds
up the sm70 GEMV ladder (1/2/4/8), so depth 4 buys 8 rows to use 5 and measured
31.5 tok/s on coding, below the 32.6 it gets with no speculation. Defaults moved
onto the ladder. Entry:
`errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md`.

**Long-context numbers here are INVALID, mine included.** A 4K prompt takes 8
chunked-prefill ticks at `max_num_batched_tokens=512`, and speculation is off on
every mixed tick (engine.py:790). The two-point slope does not cancel that at
lo=32, so 14.8/18.0 measure prefill, not decode. Long-context decode is
unmeasured; the earlier "KVSPLIT=16 is the wall" claim was a guess, withdrawn.

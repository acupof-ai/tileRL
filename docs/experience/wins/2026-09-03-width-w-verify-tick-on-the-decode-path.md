# The width-W verify tick stays on the decode path — cuda(H20), 2026-09-03

> Status: Shipped — measured twice, H20 GPU 7 and GPU 5, 2026-09-03. The tick is NaN-free on
> `main` once the split-KV guard lands, but at W=8 it still does not reproduce the unspeculated
> token stream on the 27B (4 of 8 GSM8K completions differ). See
> [errors/2026-09-03-w8-verify-tick-is-not-lossless-on-the-27b.md](../errors/2026-09-03-w8-verify-tick-is-not-lossless-on-the-27b.md).

## Context

A speculative verify tick carries W tokens per row: one committed token plus
W-1 drafts. Measured on the 27B at B=1, a W=1 tick is 10.76 ms and a W=2 tick
is 21.17 ms — a 1.97x cliff for one extra token, where a DFlash2 block of 8
needs W=8 to cost at most 2.5x a W=1 tick to break even.

The cliff is not launch overhead. At T>1 the tick falls off the decode path
in both layer families:

- **GDN.** `Model._gdn` dispatched the fused in-place decode kernel only at
  `q.shape[1] == 1`. Every wider tick took `state_gather` → `gdn_chunk_fused`
  → `state_scatter`: three gathers and two `index_put`s per layer, 240
  launches and 3.57 ms across 48 layers at depth 3, plus a 604 MB double
  write, because `gdn_chunk_fused` already takes the step-state plane as an
  operand and writes it and then the scatter copies that scratch into the
  pool.
- **Full attention.** `Backend.paged_attention` dispatched the split-KV
  flash-decoding kernel only at `s == 1`. A wider tick took the M-tiled
  prefill kernel: grid `(ceil(S/16), H, B)` = 24 blocks at B=1 on 78 SMs, one
  head per block, and the KV slice re-read per head instead of once per GQA
  group.

## What Worked

One kernel per family, generalized rather than duplicated.

**`make_gdn_decode_fused` takes T tokens.** The state column stays in
registers across the T steps, so the pool is read once and written once; the
conv window is sourced from `Windows[par] ++ this tick's qkv` exactly as
`gdn_chunk_fused` does it; and the per-chain-step state and window go
**straight into the pool's step planes** rather than into scratch the caller
then scatters. That deletes the gather/scatter pair and the double write
together — the same in-place convention the T=1 kernel already had, extended
to the whole chain. `ks=0` (a plain decode tick) aliases the step operands
onto the live planes, so the T=1 path compiles to what it compiled to before.

**`make_paged_attention_decode` takes W query tokens.** The M tile becomes
the GQA group crossed with the W chain positions, so one KV slice still
serves the whole tile and the split-KV grid `(KVSPLIT, Hkv, B)` is unchanged
— a W=8 tick reads the KV cache once, not eight times. The causal mask moves
inside the block (`p < SeqLens - SeqQLens + i % W + 1`), and taking
`SeqQLens` as an operand means rows of unequal width stay correct.
`paged_attention` routes to it while the tile fits, `s <= 8` and
`s * (H/Hkv) <= 128`; wider runs are prefill chunks and keep the M-tiled
kernel.

**The engine needed one hunk.** `_run_forward` already captured a graph per
`(B, W)`; it only padded chains to a uniform width when the graph was on.
Padding unconditionally makes every W>1 decode tick uniform-width, which is
what lets the fused kernels take one width for the whole tick on the eager
path too.

## Results

Correctness, decode kernel vs `reference.gdn_forward`, tiny shapes (B=2,
nvh=4, K=V=16), f32, GPU 2:

| | T=1 | T=4 |
|---|---:|---:|
| final state | 3.3e-09 | 2.8e-09 |
| per-chain-step states | 3.3e-09 | 7.2e-09 |
| conv window, step windows | 0.0 | 0.0 |
| out (bf16 output cast) | 3.8e-04 | 2.7e-04 |

Tighter than `gdn_chunk_fused` itself (3.1e-05 state), which is bf16-IO.

### Tick time, sweep 1 — H20 GPU 7, contended host

Graph replay of one decode tick, 20 reps, `scripts/profile_verify_replay.py
$TILERL_QWEN38_SOURCE --widths 1,2,4,8 --batches 1,8 --draft
$TILERL_QWEN38_SOURCE/model_mtp.safetensors`. GPU-busy is the CUDA-event sum
over the same replay.

| B | W | before ms (busy) | after ms (busy) | after/before | after / its own W=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 11.189 (10.99) | 11.189 (11.01) | 1.000x | 1.00x |
| 1 | 2 | 20.046 (19.60) | 17.831 (17.62) | **0.89x** | 1.59x |
| 1 | 4 | 29.701 (28.49) | 27.118 (26.04) | **0.91x** | 2.42x |
| 1 | 8 | 30.430 (29.52) | 26.949 (26.17) | **0.89x** | **2.41x** |
| 8 | 1 | 26.164 (25.44) | 26.361 (25.57) | 1.01x | 1.00x |
| 8 | 2 | 48.299 (47.32) | 42.762 (41.78) | **0.89x** | 1.62x |
| 8 | 4 | 64.290 (63.42) | 54.250 (53.25) | **0.84x** | 2.06x |
| 8 | 8 | 64.893 (64.28) | 47.205 (46.25) | **0.73x** | **1.79x** |

**The target is met.** W=8 costs 2.41x a W=1 tick at B=1 and 1.79x at B=8,
against a break-even of 2.5x; before, B=1 was 2.72x. Kernel count per tick at
W=8 B=8: 3699 -> 3009.

Two controls. W=1 is the width nothing changed at, and it reproduces to
0.0% at B=1 (11.189 both arms) and 0.8% at B=8 — the deltas at W>1 are the
treatment, not the session. And the whole BEFORE arm was measured twice, an
hour apart and across a change in host state (four neighbouring cards went
from 100% to idle between them); every row reproduced within 0.2%.

Contention cuts the right way: the last five AFTER rows ran with four
neighbouring cards at ~100% while their BEFORE counterparts ran on a quiet
host. The AFTER arm is faster anyway.

The tick is more than the replay — `engine.step()` measures 13.2 s at B=1
here because it includes the first-touch JIT, so only the replay is
comparable across arms.

### Tick time, sweep 2 — H20 GPU 5, quiet host

Same script, same widths, both arms again back to back, on the card the first
sweep never got. The host was all-zero at launch; one tenant appeared on GPU 4
during the before arm's last three rows and a second on GPU 0/2 during the
after arm, so the per-row host field records what each row actually saw.

| B | W | before ms (busy) | host | after ms (busy) | host | after/before | after / its own W=1 |
|---:|---:|---:|---|---:|---|---:|---:|
| 1 | 1 | 11.300 (11.50) | quiet | 11.696 (11.47) | 1 busy | 1.035x | 1.00x |
| 1 | 2 | 21.131 (20.69) | quiet | 18.728 (18.47) | 2 busy | **0.89x** | 1.60x |
| 1 | 4 | 31.245 (30.04) | quiet | 28.422 (27.41) | 2 busy | **0.91x** | 2.43x |
| 1 | 8 | 32.056 (31.16) | quiet | 28.257 (27.47) | 2 busy | **0.88x** | **2.42x** |
| 8 | 1 | 27.558 (26.82) | quiet | 27.888 (27.03) | 2 busy | 1.01x | 1.00x |
| 8 | 2 | 51.416 (50.46) | 1 at 17% | 45.792 (44.93) | 2 busy | **0.89x** | 1.64x |
| 8 | 4 | 68.580 (67.78) | 1 at 94% | 58.470 (57.58) | 2 busy | **0.85x** | 2.10x |
| 8 | 8 | 68.394 (67.89) | 1 at 99% | 50.386 (49.55) | 2 busy | **0.74x** | **1.81x** |

**The B=8 W=8 number did not move: 1.79x on GPU 7, 1.81x on GPU 5.** That row
is where contention should bite hardest — eight rows of work per tick leave
the least slack to hide a busy neighbour — and a 1% difference across two
cards and two host states says it was never contention-limited. B=1 W=8 is
the same story: 2.41x and 2.42x.

**Cards are not interchangeable.** GPU 5's whole before arm is 5-7% slower
than GPU 7's, measured on a *quieter* host — +5.4/5.2/5.3% at B=1 W=2/4/8 and
+6.5/6.7/5.4% at B=8. Absolute tick ms is a per-card quantity; only the
within-session ratio transfers, which is why both arms have to share a card
and a window.

One caveat on GPU 5, and it is the honest read of the B=1 row. The after
arm's W=1 was the one row that caught a busy neighbour at 96%, so it came in
3.5% above the quiet before arm's W=1 (11.696 vs 11.300) even though W=1 runs
identical code. Divide the W=8 tick by the *quiet* W=1 instead and B=1 W=8 is
2.50x rather than 2.42x — exactly at break-even rather than under it. B=8 is
unaffected either way, 1.81x within-arm and 1.83x against the quiet W=1.

## The parity gate compiled two tile widths out of three

The first sweep died in the AFTER arm: `Layout infer conflict between acc_s and
acc_s_cast`, at a `(32, 64)` fragment. The M tile is `snap(G * W)`, and the 27B
has G=6, so W=1,2 give a 16-row tile, W=4 gives 32 and W=8 gives 64. The tiny
parity case used 8 query heads over 1 KV head, which snaps to 16 and 64 and
**skips 32 entirely** — so the gate compiled two of the three tiles the real
model uses and reported green.

The tile is what the warp count has to divide: `threads = 128` over a 32-row
tile is 8 rows per warp, below the 16 an mma tile needs, so the gemm emits a
replicated fragment and the P cast can no longer match it. `threads =
32 * max(2, block_M // 16)` ties the warps to the rows; at the two widths the
sibling M-tiled kernel actually runs (16 and 64) it is the same 64 and 128 that
kernel already hard-codes. The parity case now sweeps W=4 and W=8, which is 32
and 64.

Nothing about this needs a 27B: `block_M` does not depend on `head_dim`, so a
tiny case at the third tile width would have caught it on the first run.

## Three red CUDA gates on the way in

None had ever run: CI is `TILERL_TARGET=cpu` on ubuntu and macos, so every
CUDA-only test skips there and a red CUDA gate sits red across commits.

- `test_decode_graph_matches_eager` asserted `_decode_graphs.get(1)`, a key
  that stopped existing when the cache was re-keyed to `(B, W)`. Eager and
  captured tokens were identical on clean main; the guard named a specific key
  and so failed on a healthy graph. It now asserts the cache is non-empty and,
  at the verify width, that the wide graph exists — the shape of the invariant
  rather than one key.
- `paged_attention_combine` returned an uninitialised buffer at
  `head_dim < 32`, which is every CUDA test on `config.tiny()`. A real bug,
  and the reason five speculation tests failed on main. It ships on its own
  branch: [errors/2026-09-03-combine-loop-extent-zero-at-small-head-dim.md](../errors/2026-09-03-combine-loop-extent-zero-at-small-head-dim.md).
- The decode arm of `paged_attention` handed `k_cache`/`v_cache` to the kernel
  unmigrated. Serving never hit it — `PagedKvPool` is already on device. Same
  branch as the one above.

## Found in review: NaN on one sequence length in ten

The width-W kernel emitted NaN whenever `n % 64` fell in `[1, W-1]` — a split
whose first tile sits past a low chain position's causal bound leaves both
softmax maxima at `-inf`, and the combine's `0 * NaN` carries it out. Guarded,
and `test_paged_attention_vs_naive` gains `(w, n) = (8, 65)`, which fails
against the unguarded kernel:
[errors/2026-09-03-split-kv-nan-at-verify-width.md](../errors/2026-09-03-split-kv-nan-at-verify-width.md).

The parity case had swept W while holding every `n` under one `block_N` tile,
so only `sp=0` was ever non-empty and no split geometry existed to be wrong.

One knock-on to know about: the case labelled `# prefill (T=3, ...)` now routes
to the decode kernel too (`s <= 8`), so `make_paged_attention_mma` at
`block_M=16` has no parity coverage left, and a verify width of 9-16 — legal,
`spec_depth < BLOCK_TOKENS` — still reaches it.

## Rejected: the block allreduce in the decode kernel's L2-norm

The T-loop runs `gdn_decode_fused`'s thread-0 L2 reduce once per token per
block per layer, and `gdn_chunk_fused` had already replaced that form with a
`tvm_thread_allreduce` ("a thread-0 sum was half the kernel at T=512"). Porting
it looked like the obvious remaining lever. It is not: A/B on one card in one
window, only that block changed, 27B graph replay —

| B | W | thread-0 ms | allreduce ms | ratio |
|---:|---:|---:|---:|---:|
| 1 | 1 | 11.158 | 11.140 | 0.998x |
| 1 | 2 | 17.658 | 17.663 | 1.000x |
| 1 | 4 | 26.776 | 26.787 | 1.000x |
| 1 | 8 | 26.559 | 26.608 | 1.002x |
| 8 | 1 | 26.071 | 26.048 | 0.999x |
| 8 | 2 | 40.713 | 40.876 | 1.004x |
| 8 | 4 | 52.510 | 52.177 | 0.994x |
| 8 | 8 | 44.335 | 43.800 | 0.988x |

Every row within ±1.2% with no consistent sign, against a W=1 noise floor of
0.16%. The reduce is not what the T-loop is waiting on. Reverted — the port
costs 20 lines and four locals and buys nothing. At T=512 the chunk kernel is
doing 512 of these; at T<=8 it is 8, and 8 is below the floor.

## Rule

A verify tick is a decode tick W tokens wide, not a short prefill. Dispatch
on whether the chain still fits the decode kernel's tile, not on `T == 1`.
Two arms on one card in one window; a ratio between cards means nothing when
two identical H20s differ by 5-7% on the same workload.
A guard that names a specific cache key is a guard that will one day fail on
working code; assert the shape of the invariant instead.
When a kernel picks a tile from a shape, the parity case has to sweep every
tile the real model reaches, not the smallest and the largest.
When a kernel gains a dimension, re-derive its implicit guards too: the
split-KV epilogue never said "a non-empty split holds a live key", and that
stopped being true the moment the causal bound became per-chain-position.
A lever that is worth 2x at T=512 can be worth nothing at T=8; port it and
measure it, do not reason from the entry that justified it at another size.

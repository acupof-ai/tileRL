# The backward's chunk, 16 -> 64: 1.94x — H20 sm90, 2026-09-07

> Status: ACCEPT. `_GDN_CHUNK = 16 -> 64`, one constant, `backward_secs` 80.207 -> 41.446 on
> the #192 recipe. Costs 3.5x gradient accuracy, 3.26e-06 -> 1.13e-05, 9x inside the bar.

## Context

`_GDN_CHUNK = 16` has been the backward's chunk since `51e965e`, chosen over upstream's 64 on a
precision measurement and never re-priced against a step time. It is the sequential dimension of
both python loops in `gdn_backward`: the recompute (`reference.py:927-931`) and the adjoint
(`:952-956`). At T=1280 that is 80 iterations of each, per layer, per step.

Both arms: card 6, uncontended, the #192 recipe (prompt 256, gen 1024, group 8), `--inside-gdn`,
warm step (step 2), T=1280, `wy_kernels` forward, `mlp` checkpoint segment. The only difference
between the runs is `--gdn-chunk`.

| row | C=16 secs | C=64 secs | C=16 calls | C=64 calls | ms/call 16 | ms/call 64 |
|---|---:|---:|---:|---:|---:|---:|
| `_gdn_chunk_bwd` — the adjoint | 28.612 | 7.332 | 30720 | 7680 | 0.931 | 0.955 |
| `_gdn_chunk_fwd` — the recompute | 17.193 | 4.270 | 30720 | 7680 | 0.560 | 0.556 |
| `gdn_backward` — prologue + epilogue | 5.465 | 2.283 | 384 | 384 | 14.233 | 5.946 |
| `solve_triangular` — in the recompute | 1.984 | 0.640 | 30720 | 7680 | 0.065 | 0.083 |
| unattributed | 26.952 | 26.922 | | | | |
| **`backward_secs`** | **80.207** | **41.446** | | | | |

**1.935x**, and the mechanism is in the ms/call columns: **a chunk four times larger costs the
same per call.** The adjoint is 0.955 vs 0.931 ms and the recompute 0.556 vs 0.560 — within 3%,
at 4x the work. The intra-chunk FLOPs are quadratic in the chunk length (`KK` and `QK` are n x n,
the triangular solve n x n, `A @ d` n x n x DV), so 4x the chunk is 16x the arithmetic in those
terms, and none of it shows. Both loops were pure dispatch cost.

30720 = 384 layer-calls x 80 chunks; 7680 = 384 x 20. The call counts drop exactly 4x, the
per-call time does not move, and the seconds follow the calls.

The unattributed remainder is flat in absolute terms (26.952 -> 26.922). It is **not** tape
overhead: `--inside-gdn` times exactly three `reference` functions, so every linear, rmsnorm,
attention and checkpoint in the backward falls into this residual by construction. Registry mode
at C=128 attributes 30.789 s over 21936 calls and its residual is 4.831 s, against the same work's
7.799 s attributed and 26.918 s residual under `--inside-gdn` — `backward_secs` agreeing within
2.6% either way. So the remainder is the rest of the backward, which the chunk does not touch, and
its share rising 33.6% -> 65.0% is that fixed work against a smaller total, not a regression.
[wins/2026-09-07-where-the-backward-goes.md](2026-09-07-where-the-backward-goes.md) already ranks
what is in it (`linear_fp4_frozen` 12.981 s, `checkpoint` 2.476, `linear` 1.843 at C=16).

## What it does not change

No kernel, no `saved` plumbing, no padding, no backend edit, no dispatch change. The forward is
untouched — `_WY_CHUNK = 64` already governed it, and `T % 64` still selects the forward arm
([wins/2026-09-07-a-prompt-length-picks-the-gdn-path.md](2026-09-07-a-prompt-length-picks-the-gdn-path.md)).
The 1.94x is the backward's python loop and nothing else.

Absolute seconds are not the shipped path's: the probe syncs around every timed call, which put
`backward_secs` ~11 s above #217's registry-mode reading of the same work. Both arms pay it
identically, so the ratio is the claim and the absolute numbers are not.

## The precision it costs

Same estimator `51e965e` was priced on (`gdn_backward` at chunk C against chunk 1, worst relative
over all eleven gradients, worst of three seeds), and two independent implementations of it agree:
`tests/test_ops_parity.py::test_gdn_backward_precision_tracks_the_chunk_size` prints
16 -> 3.3e-06 / 64 -> 1.1e-05, and `scripts/probe_wy_recompute_precision.py` prints
3.26e-06 / 1.13e-05.

So C=64 costs **3.5x accuracy**, 3.26e-06 -> 1.13e-05, and stays 9x inside the 1e-4 bar. That bar
is itself provisional
([errors/2026-09-07-what-gradient-error-is-acceptable.md](../errors/2026-09-07-what-gradient-error-is-acceptable.md)),
which is why this entry reports the error rather than only the ratio: if the bar moves down, this
is the number to re-read.

`51e965e`'s reasoning was not wrong — it priced the accuracy correctly and declined to pay it for
"the extra 4x of launches that 64 would have saved". What it did not have was 1.94x of a 27B
step's backward on the other side of the trade.

## Why an absolute bar cannot gate the chunk

This is the part worth not re-deriving. I first replaced the test's `_GDN_CHUNK <= 16` with an
absolute bar, expecting 128 to fail it. It passed. Measured, seeds 0/1, on the test's own fixture
shape:

| T | C=16 | C=64 | C=128 | C=256 |
|---:|---:|---:|---:|---:|
| 128 | 3.26e-06 | 1.13e-05 | 6.87e-06 | 6.87e-06 |
| 256 | 1.08e-06 | 5.38e-06 | 1.29e-05 | 3.12e-05 |
| 512 | 2.75e-06 | 5.18e-06 | 3.86e-05 | 9.88e-05 |

Two reasons, both invisible until 128 was run as a negative control. At T=128 the error
**saturates** once chunk >= T: there is one chunk, so C=128 and C=256 are the identical
computation and no threshold can order them. And growing T restores the ordering but the whole
f32 family stays inside 1e-4 — C=256 at T=512 reaches only 9.88e-5.

A 1e-4 bar therefore discriminates a *dtype* regression (the bf16 recompute arm measured 1.09e-2,
[errors/2026-09-07-what-gradient-error-is-acceptable.md](../errors/2026-09-07-what-gradient-error-is-acceptable.md))
and nothing about the chunk. The gate is now an explicit allow-list, `_GDN_CHUNK in (16, 64)`,
each value naming the entry that priced it, with the ordering assert untouched and the bar kept
as the dtype guard. Controls: 16 PASS, 64 PASS, 128 FAIL.

The old `<= 16` was right for a reason its docstring did not state — it was an allow-list written
as an inequality, not the absolute bound it disclaimed.

## Rule

Run the negative control at the value you expect to reject. The bar looked correct, the reasoning
behind it was wrong, and only a run at 128 said so. When a quantity saturates, no threshold on it
can order the inputs past that point.

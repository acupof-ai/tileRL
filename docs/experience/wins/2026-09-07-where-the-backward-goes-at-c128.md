# Where the backward goes at C=128: the GDN row is 30.48% of attributed — H20 sm90, 2026-09-07

> Status: **measurement.** Re-prices every backward-side lever on one tree at
> `_GDN_CHUNK = 128`, because the published table was measured at 16. Three arms, one job, card 0:
> the GDN row's internals, the full per-op table, and the bare `backward_secs`. The verdict is that
> no backward-side op lever clears 1.1x on the backward, and the GDN adjoint port this was scoping
> is cut on its own numbers.

## Context

The published per-op table ([where-the-backward-goes](2026-09-07-where-the-backward-goes.md))
was measured on a tree where `_GDN_CHUNK` was 16. It has been shipped at 128 since #234. Every
ROI figure derived from that table — which op to port next, what a kernel buys — is
computed against a numerator that has since fallen 6.8x and a denominator that fell with it. So
the ranking was re-measured rather than adjusted.

## The stale table and the current one

Same recipe both arms: gen 1024, group 8, warm step, registry arm of
`scripts/prof_backward_ops.py`.

| | stale | current |
|---|---:|---:|
| sha | 4bad082 | 15b6cf1 |
| `_GDN_CHUNK` | 16 (shipped; no flag existed) | 128 (`--gdn-chunk 128`) |
| `backward_secs` | 73.775 | 25.649 |
| attributed | 68.978 | 20.848 |
| peak | — | 62.60 GiB |
| `linear_attn_chunk` | 45.300 s / 65.7% | **6.355 s / 30.48%** |
| `linear_fp4_frozen` | 12.981 s / 18.8% | **4.967 s / 23.82%** |
| `linear_fp8_frozen` | 4.298 s / 6.2% | **3.144 s / 15.08%** |

Both columns are single runs. An earlier registry arm on 2cce289 read `linear_attn_chunk` 7.249 s
of 21.965 attributed — 14% above the row above it on the same code, which is the run-to-run spread
to assume before reading any two of these numbers as a change.

The fp8 row also carries a kernel now ([fp8-frozen-backward-kernel](2026-09-07-fp8-frozen-backward-kernel.md)),
so its drop is two changes and not comparable as a chunk effect. The fp4 row is untouched code
across both shas.

## Why the GDN row fell 6.8x

`_GDN_CHUNK` is the only thing that changed in the path. It was 16 at `reference.py:592`
(4bad082) and is 128 at `reference.py:591` (2cce289), moved by two commits — f0e6e71 (16→64,
#229) and 88e3764 (64→128, #234). The diff is five lines in `reference.py`: the constant and its
comment. `kernels_gdn.py` and `autograd.py` have **zero diff** across the 44 commits between the
shas.

The whole factor is per-call. Call count is **384 in both runs**, so the work is the same:
117.969 → 18.878 ms/call is **6.249x** against the 2cce289 arm, 16.551 ms/call and **7.127x**
against the arm in the table above. Both are one run against one run; the controlled figure is the
6.83x below.

Mechanism: `_GDN_CHUNK` is the sequential dimension of both python loops in `gdn_backward` —
the recompute at `reference.py:922-928` and the adjoint at `:951-956`. At T=1280 that is 80
iterations per layer-call at C=16 against 10 at C=128, and both loops were measured
dispatch-bound ([c128-backward-chunk](2026-09-07-c128-backward-chunk.md)).

Not the instrument. Identical call count means identical sync count, and the absolute
unattributed residual is 4.797 s (old) against 4.907 s (new) — 2.3% apart. The share moving from
6.5% to 18.3% is the shrinking denominator.

The finding does not rest on comparing two ad-hoc runs.
[c128-backward-chunk](2026-09-07-c128-backward-chunk.md) is a designed one-variable experiment:
four arms on card 6, same recipe, only `--gdn-chunk` differing, attributed 53.255 s at C=16 →
7.799 s at C=128, **6.83x**.

## What that invalidates

The 55/35/10 split of the GDN row — adjoint 55-57%, recompute 31-35%, prologue/epilogue 10-13%
([inside-the-gdn-backward](2026-09-07-inside-the-gdn-backward.md)) — was measured at C=16. It is
superseded. Measured at C=128, `--inside-gdn`, warm step 2, `backward_secs` 24.792, attributed
7.188 s over 11904 calls:

| GDN internal | secs | share of GDN | calls | ms/call | C=16 bracket |
|---|---:|---:|---:|---:|---:|
| `_gdn_chunk_bwd` (the adjoint) | **3.102** | **43.16%** | 3840 | 0.808 | 55-57% |
| `_gdn_chunk_fwd` (recompute, excl. the solve) | 1.864 | 25.94% | 3840 | 0.485 | 31-35% |
| `gdn_backward` (prologue + epilogue) | **1.784** | **24.82%** | 384 | 4.645 | 10-13% |
| `solve_triangular` (in recompute) | 0.438 | 6.09% | 3840 | 0.114 | — |

The prologue and epilogue is the part that did not benefit from the chunk increase: the two loops
run 10 iterations instead of 80 while it runs once per layer-call either way, so its share roughly
doubled while the adjoint's fell 12 points. This arm's 7.188 s total is within 0.8% of the
registry arm's 7.249 s row, which is what shows both arms measure the same work.

The general point: a share is not a property of an op. Both terms moved here, and the fp8 row's
share rose from 6.2% to 14.4% while its seconds fell from 4.298 to 3.162.

## The lever table

C=128, sha 15b6cf1, warm step 2, exclusive seconds over **20.848 s** attributed
(`backward_secs` 25.649). The step-gain column divides against this sha's bare
`--no-instrument` backward, **22.145 s** (warm step 2, peak 62.62 GiB; step 1 was 22.571), so it
is a gain on the BACKWARD and not on a GRPO step: these arms run no
rollout at all — `train_secs` 25.728 against `backward_secs` 25.649 is a 0.08 s optimizer — so
the step denominator is unmeasured on this tree and the rollout share is unknown.

| op | secs | share of attributed | realistic gain at a 1.4x-class op speedup | resulting BACKWARD gain |
|---|---:|---:|---:|---:|
| `linear_attn_chunk` | 6.355 | 30.48% | 1.851 | 1.091x |
| `linear_fp4_frozen` | 4.967 | 23.82% | 1.447 | 1.070x |
| `linear_fp8_frozen` | 3.144 | 15.08% | 0.916 | 1.043x |
| `checkpoint` | 2.483 | 11.91% | 0.723 | 1.034x |
| `linear` | 1.828 | 8.77% | 0.532 | 1.025x |
| `rmsnorm` | 0.829 | 3.97% | 0.241 | 1.011x |
| `attention` | 0.572 | 2.74% | 0.167 | 1.008x |
| `silu_mul` | 0.322 | 1.54% | 0.094 | 1.004x |
| `rmsnorm_f32` | 0.136 | 0.65% | 0.040 | 1.002x |
| `rope` | 0.084 | 0.40% | 0.024 | 1.001x |
| `slice` | 0.082 | 0.39% | 0.024 | 1.001x |
| `add` | 0.040 | 0.19% | 0.012 | 1.001x |
| `reshape` | 0.008 | 0.04% | 0.002 | 1.000x |
| `embedding` | 0.000 | 0.00% | 0.000 | 1.000x |

Per-call for the three largest rows: `linear_attn_chunk` 16.551 ms over 384 calls,
`linear_fp4_frozen` 2.352 ms over 2112, `linear_fp8_frozen` 1.686 ms over 1864.

**No backward-side op lever clears 1.1x on the backward.** The top row is 1.091x at the 1.4x
class, and it is the whole GDN row — all three of its internals, not a phase. Every other row is
at or under 1.07x. The `--inside-gdn` split prices the adjoint alone, the part a port would
replace first, at 3.102 s: **1.043x**, the same as the fp8 dX kernel bought for one gradient
instead of six.

The two gain columns are computed once arm 1's split lands, because a port reaches part of a
handler and not the row. 1.4x is the class **measured** for the fp8 dX kernel today — 1.411x on
the op, from both arms in one process
([fp8-frozen-backward-kernel](2026-09-07-fp8-frozen-backward-kernel.md)) — used as the one
frozen-dX data point on this card, not as an assumption about these ops.

Why the eager adjoint is slow, read from the code: `reference._gdn_chunk_bwd` is 58 lines
(`reference.py:629-686`) containing 15 matmuls, 2 einsums, 9 reductions, 18 permutes and 14
transposes. The layout traffic outnumbers the matmuls two to one.

The instrument's cost, both arms measured on this sha: registry 25.649 − 22.145 bare = **3.504 s**;
`--inside-gdn` 24.792 − 22.145 = **2.647 s**. The two wrappers sit at different depths and their
call counts measure different things, so neither bounds the other and no per-call cost is
recoverable across them. The port note recorded 27.264 − 23.239 = 4.025 s on its own tree.

## Traps

The first launch died rc=143 because the arms script called `pod_run_claim` on its own bash pid.
`card_claim.py` refuses a shell outright — "pid ... is a shell, not the job" — and names the python
descendant to claim instead, so the poll ended at once rather than after `DEVICE_WAIT`, and
`pod_run_claim` fell to its exit branch. The wrapper's own trap then killed the job: rc=143.
**rc=143 is indistinguishable from an external kill at the log level** — the same code, opposite
causes, and neither legible without reading `/work/pod_run_<name>.out`, which is where the refusal
sentence actually is.

Removing that call replaced one bug with another: `pod_run.sh` claims the WRAPPER pid, so a sweep
whose arms are separate pythons loses its claim when the first arm exits while the card stays
busy — `card_claim.py status` then reports the card as a reclaimable ORPHAN, and a peer about to
launch caught it. The launcher documents the rule and exports `pod_run_claim` so a multi-arm
wrapper can call it per arm; the correct call is on each arm's python pid.

Re-claiming by hand hit a third one: `nvidia-smi --query-compute-apps` reports HOST pids while
`card_claim.py` reads the container's `/proc`, so a claim taken on the pid the obvious instrument
gives is stale the moment it is written. The container-visible pid from `ps -ef` inside
`sglang-test` is the one that holds.

## The prologue and epilogue is 92 eager launches, not bandwidth

Read after the split, because 1.784 s at 4.645 ms/call over only 384 calls is the one row whose
cost is not explained by its FLOPs. `reference.py:858-998` is 141 lines, 139 of them outside the
two chunk loops, and they issue **92 eager tensor ops** per call: 12 `_f32` casts, 14 tensor
constructions, 9 reductions, 9 reshapes, 6 `zeros`, 5 `zeros_like`, 5 sigmoids, 3 silus, 3 rsqrts.

Bandwidth does not account for it. The largest structure is the conv1d recompute at `:899-903` —
a 4-tap python loop, each tap padding and multiplying the whole `[b, t, 3*dim]` qkv, 50 MB f32 a
pass and about 600 MB for the loop — and that loop plus both `repeat_interleave` materialisations
is **0.32 ms at 2 TB/s against a measured 4.645 ms**. 92 launches of eager dispatch fits; a
bandwidth story does not.

So the available win here is fusing many small ops, each fused group needing its own gradient
gate. Removing all 1.784 s caps at 1.084x on the backward. Nothing is shipped for it.

## Rule

A share is only meaningful with its denominator's config. Re-read a numerator before scoping work
against it — the top row here was still being quoted at 65.7% after the shipped tree made it
33.0%.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-07 | 15b6cf1 | H20 pod (GPU 0) | cuda/sm90 | Qwen3.8-27B NVFP4, GRPO backward | — | — | — |

Measurement entry; no runtime change, so the three serving columns do not apply. The metric is
`backward_secs` at `--gdn-chunk 128`, warm step 2: **22.145 bare**, 25.649 registry-instrumented,
24.792 `--inside-gdn`, peak 62.62 GiB. Three arms in one job on card 0, sequential.
Raw artifacts on the pod: `/work/levers2.log`, `/work/lev_inside.json`, `/work/lev_registry.json`,
`/work/lev_bare.json`.

# The GDN layer around the WY core is two kernels, not sixty torch ops — H20, 2026-09-03

> Status: Shipped. sm90 default flipped to the WY path, `TILERL_GDN_WY` deleted.
> Gate 1.15x, measured 1.29 / 1.29 / 1.28x on a quiet host.

## Context

The chunkwise-WY prefill core landed faster than fla — 137.4 us/layer against
145.9, where the shipped serial `gdn_chunk_fused` costs ~1400 — and the
27B model-level prefill fell 233.9 → 189.9 ms of GPU-busy, 1.23x. The harness
moved 2238.6 → 2324.6 tok/s, 1.036x.

The gap is not the core. The WY path reached the core through
`reference.gdn_forward`, whose conv1d, SiLU, L2-norm, gate, beta and gated
RMSNorm are torch ops: kernel count went 3827 → 6755, and prefill spends
0.45 s of wall against 0.19 s of GPU-busy. Prefill is host-bound, so 2900
extra launches cost more than 44 ms of saved GPU time bought.

## What Worked

`gdn_chunk_fused` already contains that math; it was split rather than
rewritten. `gdn_prep` emits exactly what the WY kernels consume — q and k
SiLU'd, L2-normed and scaled, v SiLU'd, the cumsum-ready log gate, sigmoid
beta, and the next conv window — and `gdn_post` is the epilogue, gated RMSNorm
times SiLU(z), which had been `rmsnorm` + two casts + `silu_mul`.

Both land as CPU cells first (`kernels.make_gdn_prep` / `make_gdn_post`,
portable f32, no shared memory, reductions serial into a fragment scalar), with
an sm90 override for prep (`kernels_gdn.make_gdn_prep_bf16`: one thread per
head column, the two L2 sums by block allreduce, bf16 out for the WY gemms).
`gdn_post` needs no second schedule — the same kernel is registered with a bf16
IO argument on sm90.

That makes the WY path real on the GPU-less machine: `linear_attn_chunk` now
routes full-length T>1 rows through `Backend._gdn_chunk_wy` on every target,
and the cell without a WY schedule (cpu, metal) uses `reference.gdn_chunk_core`
as the middle stage. `test_gdn_chunk_fused_parity*` stopped being the tautology
its docstring admits to off sm90.

Glue counted by aten dispatch, one GDN layer at the 27B shape (B=1, T=512,
16 key heads, 48 value heads, D=128), the chunkwise core subtracted from both:

| path | torch ops per layer |
|---|---:|
| `reference.gdn_forward` (the old WY route) | 48 |
| `Backend._gdn_chunk_wy` | 2 |

48 is a floor: on CPU the eleven `_f32` casts are no-ops, on CUDA six of them
are real kernels. Working back from the measured 2928-kernel gap over 48 GDN
layers, the old route cost 75 launches a layer, of which 68 were glue; the new
one costs 15.

## Rule

A kernel that beats SOTA inside a torch-eager layer has not shipped anything.
Count launches per layer before believing an end-to-end ratio — on this model
prefill is host-bound, and the glue outweighs the kernel it wraps.

## Results

Every row says how it was obtained. A derived row read as measured is how a
bound turns into a fake measurement.

Host for the timing rows: H20, GPU 4, `scripts/profile_prefill.py` and
`scripts/bench_harness.py`. GPUs 2, 3, 5, 6, 7 idle; GPU 0 (50977 MiB) and
GPU 1 (15973 MiB) held single-card jobs by other tenants throughout, on their
own cards. The shipped-serial arm was re-measured in the same window rather
than quoted, and it reproduces the recorded quiet-host baseline to 0.2%
(233.8 vs 233.9 ms, 3827 vs 3827 kernels, 2242.3 vs 2238.6 tok/s) — which is
what licenses the ratio.

| metric | how | shipped serial | WY, torch glue | WY + prep/post |
|---|---|---:|---:|---:|
| prefill GPU-busy, ms (64 layers, 512 tok) | measured | 233.8 | 189.9 | **176.4** |
| kernels per prefill | measured | 3827 | 6755 | **3925** |
| GDN launches per layer | derived from the row above | 14 | 75 | 16 |
| harness prefill 512, tok/s | measured | 2242.3 | 2324.6 | **2891.7** |
| harness prefill 2048, tok/s | measured | 2221.7 | — | **2854.2** |
| harness prefill 8192, tok/s | measured | 2147.1 | — | **2732.8** |
| WY core, us/layer (fla 145.5) | measured | ~1400 | 137.4 | 121.8 |

Ship gate was 1.15x against the recorded 2238.6 / 2215.9 / 2142.4 tok/s:

| depth | vs recorded baseline | vs same-session serial |
|---|---:|---:|
| 512 | 1.292x | 1.290x |
| 2048 | 1.288x | 1.285x |
| 8192 | 1.276x | 1.273x |

**The prediction landed, and where it missed.** The launch-count argument rested
on ~3875. Actual 3925 — **+50, +1.3%**. Per layer that is 16 launches, not the
15 predicted: one call a layer I did not account for. The direction and the
magnitude hold, and the 1.77x launch inflation that made the fast core read as
1.036x is gone (3925/3827 = 1.026x).

`gdn_prep` costs 3.34 ms over 48 launches and `gdn_post` 5.35 ms over 48 —
8.7 ms of GPU for what had been ~36 ms of torch glue. `rmsnorm_fused` drops
209 → below the top 8 and `silu_mul` 112 → 64 launches, both absorbed by
`gdn_post`. The WY path now also beats fla's own chunked route on this model
end to end: 176.4 ms and 3925 kernels against 193.4 ms and 6899.

**Caveat on the 8192 row.** Its first measurement carried 27.0% spread where
512 and 2048 carried 0.1%, so the absolute 2732.8 is soft. Re-running that
depth alone caught the reason: the neighbouring tenant's five-GPU job restarted
mid-run and took GPU 4, and both arms collapsed together — serial 964.8 and
967.2 tok/s, WY 1240.5 and 1240.4. The absolute numbers from that pass are
discarded. The *ratio* survived contention unchanged, which is the number the
gate reads: **1.273x, 1.286x, 1.282x across three independent runs**, two of
them on a shared card. Treat 2732.8 as approximate and 1.27-1.29x as solid.

sm90 parity, `scripts/probe_gdn_wy.py` on H20, gate 1e-2 relative:

| check | out | state | window |
|---|---:|---:|---:|
| (b2) `linear_attn_chunk` (WY) vs `reference.gdn_forward` | 5.9e-3 | 7.7e-3 | 0 |
| (b) `_gdn_wy_core` vs `reference.gdn_chunk_core` | 6.6e-3 | 6.1e-3 | — |
| (b) `_gdn_wy_core` vs fla `chunk_gated_delta_rule` | 3.5e-3 | 4.8e-3 | — |

(b2) is the gate for the two new cells — it is the only thing that runs
`make_gdn_prep_bf16` and the bf16 `gdn_post` at all. The bf16-IO error is three
orders above the f32 cells and still inside 1e-2; the same shape appears in
`test_gdn_chunk_fused_parity_full_scale`, where the bf16 kernel lands at 1.2%
against a 5% tolerance.

Per-kernel parity of the WY core against fla 0.5.2: `kkt` 4.6e-4 max abs,
`solve_tril` 4.9e-4, `w` 2.4e-4, `u` 0, `h` 9.8e-4, `V_new` 9.8e-4, final state
4.7e-4, `o` 2.3e-4. State-scan known-answer rows exact.

Parity on the GPU-less machine, `Backend.linear_attn_chunk` vs
`reference.gdn_forward`'s per-step scan, tiny model at full input scale:
cpu out 2.1e-6 / state 4.3e-7 / window 0, metal 7.5e-7 / 4.9e-7 / 0.
`TILERL_TARGET=cpu uv run pytest -q`: 170 passed.

The 137.4 → 121.8 us/layer on the core is not a schedule change:
`gdn_chunk_core_tl` cast q, k, v and beta on every call and `gdn_prep` now
emits them already cast, leaving only the 3.0 us state cast inside the core.

## What did not change

The serial `gdn_chunk_fused` stays and is still reachable: speculative verify
(`keep_steps`), ragged rows, and any length that is not a whole multiple of the
64-token chunk. That is a capability boundary, not a second path to the same
answer.

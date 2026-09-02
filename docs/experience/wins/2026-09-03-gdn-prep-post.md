# The GDN layer around the WY core is two kernels, not sixty torch ops — H20, 2026-09-03

> Status: pending-remote (CPU/Metal parity measured here; every H20 number below
> is unmeasured — no card was free, and a prefill number taken while five
> neighbouring GPUs are saturated is not comparable to the baseline it is a
> ratio against)

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

| metric | shipped serial | WY, torch glue | WY + prep/post |
|---|---:|---:|---:|
| WY core, us/layer (fla 145.9) | ~1400 | 137.4 | 137.4 |
| prefill GPU-busy, ms (64 layers, 512 tok) | 233.9 | 189.9 | pending-remote |
| kernels per prefill | 3827 | 6755 | ~3875 predicted |
| GDN launches per layer | 14 | 75 | 15 |
| harness prefill 512, tok/s | 2238.6 | 2324.6 | pending-remote |
| harness prefill 2048, tok/s | 2215.9 | — | pending-remote |
| harness prefill 8192, tok/s | 2142.4 | — | pending-remote |

The two totals and the 2928-kernel gap are measured (`scripts/profile_prefill.py`,
2026-09-02); the per-layer rows are read off the code against that gap, and the
predicted total follows from them.

Per-kernel parity of the WY core against fla 0.5.2, measured 2026-09-02 on H20
and unchanged by this entry: `solve_tril` 4.9e-4 max abs, `w` 2.4e-4, final
state 4.7e-4, `o` 2.3e-4.

Parity measured here: `Backend.linear_attn_chunk` (WY path) vs
`reference.gdn_forward`'s per-step scan, tiny model at full input scale,
relative max-abs — cpu out 2.1e-6 / state 4.3e-7 / window 0, metal 7.5e-7 /
4.9e-7 / 0. Gate is 1e-2. `TILERL_TARGET=cpu uv run pytest -q`: 170 passed.

Ship gate, unrun: harness prefill at or above 1.15x of 2238.6 / 2215.9 / 2142.4
tok/s at 512 / 2048 / 8192, `CUDA_VISIBLE_DEVICES=<free> python3
scripts/bench_harness.py --suite prefill --source /work/Qwen3.8-27B-NVFP4`.
Until it runs, sm90 keeps the serial kernel and the WY path stays behind
`TILERL_GDN_WY=1`; `scripts/probe_gdn_wy.py` section (b2) is the sm90 parity
check for the two new cells.

# The GDN layer around the WY core is two kernels, not sixty torch ops — H20, 2026-09-03

> Status: pending-remote for the ship gate. sm90 compile and parity ARE measured
> (H20 GPU 2, idle); throughput is not — six of eight cards on that host were
> saturated by another tenant, and a prefill number taken beside them is not
> comparable to the quiet-host baseline it would be a ratio against.

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

| metric | how | shipped serial | WY, torch glue | WY + prep/post |
|---|---|---:|---:|---:|
| WY core, us/layer (fla 145.5) | measured | ~1400 | 137.4 | 121.8 |
| prefill GPU-busy, ms (64 layers, 512 tok) | measured | 233.9 | 189.9 | **pending-remote** |
| kernels per prefill | measured | 3827 | 6755 | **pending-remote** |
| GDN launches per layer | derived from the row above | 14 | 75 | 15 |
| kernels per prefill | predicted from the row above | — | — | ~3875 |
| harness prefill 512, tok/s | measured | 2238.6 | 2324.6 | **pending-remote** |
| harness prefill 2048, tok/s | measured | 2215.9 | — | **pending-remote** |
| harness prefill 8192, tok/s | measured | 2142.4 | — | **pending-remote** |

The 137.4 → 121.8 us/layer is not a schedule change: `gdn_chunk_core_tl` cast
q, k, v and beta to bf16 on every call, and `gdn_prep` now emits them already
cast. Only the state cast is left inside the core, at 3.0 us. Taken on an idle
GPU 2 while six neighbours were saturated, so treat it as a floor, not a
headline.

sm90 parity, `scripts/probe_gdn_wy.py` on H20 GPU 2, gate 1e-2 relative:

| check | out | state | window |
|---|---:|---:|---:|
| (b2) `linear_attn_chunk` (WY) vs `reference.gdn_forward` | 5.9e-3 | 7.7e-3 | 0 |
| (b) `_gdn_wy_core` vs `reference.gdn_chunk_core` | 6.6e-3 | 6.1e-3 | — |
| (b) `_gdn_wy_core` vs fla `chunk_gated_delta_rule` | 3.5e-3 | 4.8e-3 | — |

(b2) is the gate for the two new cells — it is the only thing that runs
`make_gdn_prep_bf16` and the bf16 `gdn_post` at all. Both compile. The bf16-IO
error is three orders above the f32 cells and still inside 1e-2; the same shape
appears in `test_gdn_chunk_fused_parity_full_scale`, where the bf16 kernel
lands at 1.2% against a 5% tolerance.

Per-kernel parity of the WY core against fla 0.5.2, same run: `kkt` 4.6e-4 max
abs, `solve_tril` 4.9e-4, `w` 2.4e-4, `u` 0, `h` 9.8e-4, `V_new` 9.8e-4, final
state 4.7e-4, `o` 2.3e-4. The state-scan known-answer rows are exact
(`e_last` at g=-0.25 gives 1.125e-07 against 1.125e-07).

Parity on the GPU-less machine, `Backend.linear_attn_chunk` (WY path) vs
`reference.gdn_forward`'s per-step scan, tiny model at full input scale:
cpu out 2.1e-6 / state 4.3e-7 / window 0, metal 7.5e-7 / 4.9e-7 / 0.
`TILERL_TARGET=cpu uv run pytest -q`: 170 passed.

Ship gate, unrun and waiting on a quiet host: harness prefill at or above 1.15x
of 2238.6 / 2215.9 / 2142.4 tok/s at 512 / 2048 / 8192, plus
`scripts/profile_prefill.py` for the real kernel count against the ~3875
prediction. Until it runs, sm90 keeps the serial kernel and the WY path stays
behind `TILERL_GDN_WY=1`.

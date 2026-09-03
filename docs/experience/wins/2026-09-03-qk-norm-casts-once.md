# The q/k norm casts once — H20 sm90 card 6, 2026-09-03

> Status: Shipped (correctness); perf `pending-remote`

## Context

`q_norm` and `k_norm` feed rope and then the bf16 KV pool. On sm90 the norm
kernel stored bf16, so the value was rounded there and again at the pool — two
roundings where one is needed. This is the arm the 53 MMLU answer flips came
from ([errors/2026-09-03-unfused-prelude-double-rounds.md](../errors/2026-09-03-unfused-prelude-double-rounds.md)),
and it is `cli.py`'s default and the arithmetic the training tape runs backward
through.

## What Worked

`make_rmsnorm_fused` takes `out_dtype`; the two q/k call sites go through a new
`Backend.rmsnorm_f32` that resolves `rmsnorm_fused_f32`. One launch either way,
so the output dtype is the only variable between arms.

Measured through the shipped call path, 64 tokens x 4 KV heads, against a dense
f64 norm+rope (`reference.attn_prelude`):

| | |
|---|---:|
| elements the change moves | 3163 / 65536 |
| mean \|err\| before (bf16 store) | 2.049e-03 |
| mean \|err\| after (f32 store) | 1.009e-03 |
| **ratio** | **2.0314** |
| closer to exact after / before | **3163 / 0** |

2.0314 matches the 2.0007 that separated the discrete prelude from `attn_prep`,
which is what identifies this store as the extra rounding rather than something
correlated with it.

**The same measurement read 1.057 first**, as a mean over all 65536 elements:
95.2% are identical in both arms and contribute the same error to each, diluting
the ratio toward 1.0. That printed `PREDICTION FAILED: casting once is the wrong
fix` — the opposite of the truth. The tell was a max ratio of 1.946 beside a mean
ratio of 1.057, which cannot both describe one effect.

## Rule

Report every mean with the n it was taken over. `2.048e-03 over 3165` and
`9.272e-04 over 65536` cannot be compared by accident; two bare means invite it.

A new backend op is not a new kernel plus a call site. `_TapeBackend.__getattr__`
returns any op absent from `_BWD` raw and records **nothing** — measured 1 tape
entry for `rmsnorm` and 0 for an unregistered twin, with forward parity, the CPU
twin and the sm90 oracle all still green while `q_norm` and `k_norm` silently
stop learning. Registering the backward is the load-bearing line, and it has no
natural gate.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---|---|---|
| 2026-09-03 | 03956b0 | H20 card 6 | cuda sm90 | 27B NVFP4 | pending-remote | pending-remote | pending-remote |

**Perf is `pending-remote`, deliberately, and one number here is invalid.** A
kernel-level A/B read **226x** for the f32 path — that measurement compared one
fused launch against `rmsnorm_partial` plus a freshly JIT-built `rmsnorm_apply`,
conflating output dtype, launch count and compile time inside the timed region.
It is not an upper bound either, because a bound built from a conflated
measurement bounds nothing. The f32 fused kernel exists precisely so a clean
one-launch-vs-one-launch A/B is possible; nobody has run it.

The open question is prefill, not decode: an f32 q/k norm output doubles that
tensor's store and the rope load, and prefill norms the whole prompt while decode
touches a handful of rows. Engine-level prefill tok/s before/after is the
measurement that settles it.

## Scope

Two call sites, not the shared kernel.

| site | consumer | changed |
|---|---|---|
| `q_norm`, `k_norm` (`model.py:240-241`) | rope -> bf16 KV pool | **yes** |
| `input_norm` (`model.py:197,271`) | fp4/fp8 GEMM | no — the GEMM's cast rounds harder |
| `post_attn_norm` (`model.py:338`) | fp4/fp8 GEMM | no — same |
| `final_norm` (`model.py:391`) | quantized lm_head | no — **config-dependent** |
| `dflash2.py:236-237` | `torch.cat` + `_attend`, no pool | no — no second rounding to remove |

`final_norm` is the one to revisit: it is unchanged because this checkpoint is
`tie_word_embeddings=False` and `load_hf` quantizes every spec key except
`embed_tokens`, so its output meets an fp8 cast. A tied head is `embed_tokens`,
which is never quantized, and a bf16 checkpoint has no fp8 cast at all — in
either case the bf16 intermediate would survive and this row would need
remeasuring. Nobody has measured whether it moves a logit even here.

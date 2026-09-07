# Board question: what relative gradient error is acceptable? — 2026-09-07

> Status: OPEN, for ckl. No work proceeds on it. Blocks the GDN adjoint kernel port
> (55% of the backward's largest row); does not block the C=64 f32 chunk change.

## The question

Every gate we have says a backward is correct at `rel < 1e-4` or so. Nothing behind that
number is measured against training outcomes — I set 1e-4 by fiat when scoping the C=64
recompute, and the shipped `_GDN_CHUNK = 16` was chosen on a *relative* argument (16 rounds
better than 64) rather than an absolute one.

The port that follows needs an absolute answer, because bf16 stage outputs land at **1e-2**,
two orders past anything we have shipped, and no subset of them is better.

## What is measured

`scripts/probe_wy_recompute_precision.py`, CPU, f32 torch, estimator identical to the one the
C=16 verdict was priced on ([wins/2026-08-29-chunked-gdn-backward.md](../wins/2026-08-29-chunked-gdn-backward.md)):
`gdn_backward` at chunk C against chunk 1 (the serial scan), worst relative error over all
eleven gradients, worst of seeds 0/1/2.

| arm | worst rel |
|---|---:|
| C=16 f32 — shipped | 3.26e-06 |
| C=64 f32 | 1.13e-05 |
| C=64 with bf16 stage outputs | **1.09e-02** |

bf16 costs **965x at equal chunk**. Per-tensor, one rounded at a time:

| rounded to bf16 | worst rel |
|---|---:|
| `d` = U − W·S, the scan's `V_new` | 1.10e-02 |
| `s` = the chunk entry state, the scan's `h` | 7.83e-04 |
| `M` = (I+L)⁻¹, `gdn_solve_tril`'s `Ai` | 7.80e-04 |
| `W` = M·(βe·K), the scan's `w` | 1.26e-04 |

All four fail a 1e-4 bar individually, so there is no reduced version of the plan.

That arm is a **simulation**: it rounds the four tensors inside the f32 reference rather than
reading them from the kernels. It bounds the dtype effect and nothing more — `gdn_solve_tril`'s
block products run at tf32 (`kernels_gdn.py:213`), so the real kernel path is no better than
1.09e-2.

## Why this is not one PR's problem

**The upstream adjoint kernels emit bf16 gradients.** `example_chunk_delta_bwd.py:556-560` runs
`input_dtype=bfloat16, output_dtype=bfloat16, accum_dtype=float32` at `chunk_size=64`, so the
gradients themselves come out in bf16 even though the accumulation and the state are f32 there.
Porting it puts a bf16 rounding on the 55% of `linear_attn_chunk`'s 45.300 s that is the adjoint
proper ([wins/2026-09-07-where-the-backward-goes.md](../wins/2026-09-07-where-the-backward-goes.md)).
That is a *narrower* exposure than the recompute arm measured above — one rounding on the output
rather than four on the inputs — so the 1.09e-2 figure is not a prediction for the port. It is
the reason the port needs an absolute bar before it is written, since nothing here says where
one output rounding lands.

**The forward already carries bf16 state, and upstream's does not.** `backend.py:1250-1253`
casts the state to bf16 for the scan's gemm operand and `gdn_state_scan` declares its
`initial_state` and `h` bf16 (`kernels_gdn.py:367-368`); upstream's own backward asks for
`state_dtype=float32`. So on the WY path our forward's recurrence rounds at bf16 every chunk,
and our f32 reference backward is the exact adjoint of a *more accurate forward than the one
that ran*. Whatever we decide, "the backward must be f32" and "the forward is f32" are not the
same claim on this path — and the pair may already be inconsistent.

## What would answer it

Two candidates, in increasing cost:

1. **The standard bf16-training argument.** bf16 gradients are the industry default for
   pretraining and fine-tuning at this scale; if a 1e-2 relative error on a LoRA update is
   inside the noise of the optimizer's own stochasticity, the bar belongs near 1e-2, not 1e-4.
   Cheap, but it is an argument from convention, not a measurement on this model.
2. **A two-arm RL run.** Same seed, same prompts, `_GDN_CHUNK` and the recompute dtype the only
   difference, compare the reward trajectory over enough steps to separate the arms from
   seed noise. Costs card time and needs a stated number of steps before it starts.

Neither has run. The bar stays 1e-4 in the meantime, which is the conservative choice and the
reason the bf16 recompute was rejected rather than shipped.

## Rule

A correctness bar is a claim like any other. `rel < 1e-4` decided a port's fate before anyone
measured what error the training run can absorb — and the same number would have rejected the
forward path we already ship, which rounds its state at bf16.

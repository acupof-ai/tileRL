# The backward is GDN and the frozen linears — H20 sm90, 2026-09-07

> Status: **measurement.** Two candidate levers are refuted, two real ones named.

## Verdict

Three ops are **90.7%** of a 68.98 s attributed backward:

| op | secs | share of attributed | what it is |
|---|---:|---:|---|
| `linear_attn_chunk` | **45.300** | **65.7%** | GDN backward, torch-eager |
| `linear_fp4_frozen` | **12.981** | **18.8%** | dX through a frozen fp4 weight, already a kernel |
| `linear_fp8_frozen` | **4.298** | **6.2%** | same op, fp8 — falls through to eager |

The two candidates that had been in the air for two days are both nothing:

| candidate | measured | share |
|---|---:|---:|
| the 64 recomputed MLP forwards (`checkpoint`) | 2.476 s | **3.6%** |
| a TileLang full-attention backward to replace the eager one | 0.565 s | **0.8%** |

A TileLang attention backward, if it were **free**, would buy 0.8% of the backward. The GDN
backward alone is 61.4% of `backward_secs`.

Its share of a whole GRPO step is **not computable from this run**: this probe builds with
`decode_graph=False` (`prof_backward_ops.py:237`), so its rollout is eager and its step time is
not the shipped path's — the reason stated under Limitations below. An earlier version of this
line read "roughly 32% of a whole GRPO step", against #192's 133 s step, which those same
Limitations say is not comparable. On the shipped tree the step is 85.617 s and the whole
forward+backward bucket is 26.1% of it
([the step is 74% rollout](2026-09-07-the-step-is-74-percent-rollout.md)), so no single op in
this table can reach 32% of a step. The **shares of `backward_secs`** are what this entry
establishes.

Eager handlers are **50.567 s (73.3%)** of attributed against 15.795 s (22.9%) in TileLang
kernels and 2.616 s (3.8%) in views.

## Context

A GRPO step at gen 1024, group 8, is 133 s and `backward_secs` is **69.7** of it — 52%
([#192](2026-09-06-one-grpo-step-is-54-percent-backward.md)). That figure is one bucket:
`rl_step` times the whole `tape.backward` call and nothing splits it, so "backward is the
lever" has been true and unactionable for two days. Row 28 of the task board carried a per-op
profile since 2026-09-06 with nobody running it.

Two candidate levers were already in the air without evidence: the 64 MLP forwards recomputed
inside the backward (`autograd.checkpoint` with `recompute=True`), and a TileLang attention
backward to replace the torch-eager one. This table is what decides between them, or names a
third.

## Method

`scripts/prof_backward_ops.py` wraps `autograd._BWD` — the handler registry — so every
backward op is timed by name. `Tape.backward` resolves that dict at dispatch, and a sub-tape
from `checkpoint` resolves the same dict, so a segment's inner ops are attributed to their own
names with no second hook.

**Exclusive time.** `checkpoint`'s handler replays its segment through the registry, so an
elapsed-time wrapper charges the inner `linear` and `rmsnorm` to both their own rows and
`checkpoint`'s. Measured on tiny: **12 nested calls**, `checkpoint` reading 24.20 ms inclusive
against **2.10 ms** exclusive, with the shares summing past 100%. Each frame subtracts what
its callees took; the tiny table then sums to exactly 100.0%.

**Kernel or eager, read rather than guessed** (`backend.py`): `rmsnorm_bwd` calls
`rmsnorm_rstd` + `rmsnorm_bwd_x` (:536), `linear_bwd` calls `gemm_nn` (:605),
`linear_frozen_bwd` calls `linear_fp4_bwd` when the scale block is 16 **and `not fp8`**
(:1118) — the fp8 arm falls through to `reference.linear_frozen_bwd` (:1136), which is why
`linear_fp8_frozen` is labelled eager. Everything else carries
`# ponytail: torch-eager backward`.

Two rows carried the wrong label in the first draft of this entry: `linear_fp8_frozen` was
written as a kernel because it shares `linear_frozen_bwd` with the fp4 arm. It shares the
function, not the kernel — the `not fp8` guard sends it elsewhere.

## Results

Step 2, warm. Exclusive seconds, one GRPO step at gen 1024 group 8.

| op | secs | share | calls | ms/call | kind |
|---|---:|---:|---:|---:|---|
| `linear_attn_chunk` | 45.300 | 65.67% | 384 | 117.970 | eager (GDN) |
| `linear_fp4_frozen` | 12.981 | 18.82% | 2112 | 6.146 | kernel: linear_fp4_bwd (sm90) |
| `linear_fp8_frozen` | 4.298 | 6.23% | 1864 | 2.306 | eager (no fp4_bwd for fp8) |
| `checkpoint` | 2.476 | 3.59% | 512 | 4.837 | recompute + the segment's own ops |
| `linear` | 1.843 | 2.67% | 7952 | 0.232 | kernel: gemm_nn |
| `rmsnorm` | 0.835 | 1.21% | 1032 | 0.809 | kernel: rmsnorm_rstd + rmsnorm_bwd_x |
| `attention` | 0.565 | 0.82% | 128 | 4.414 | eager |
| `silu_mul` | 0.321 | 0.46% | 512 | 0.626 | eager |
| `rmsnorm_f32` | 0.136 | 0.20% | 256 | 0.531 | kernel: rmsnorm_rstd + rmsnorm_bwd_x |
| `slice` | 0.092 | 0.13% | 1408 | 0.066 | view |
| `rope` | 0.082 | 0.12% | 256 | 0.322 | eager |
| `add` | 0.039 | 0.06% | 5000 | 0.008 | view |
| `reshape` | 0.008 | 0.01% | 512 | 0.016 | view |
| `embedding` | 0.000 | 0.00% | 8 | 0.010 | eager |

- `backward_secs` (`rl_step`'s own timing, no per-handler sync): **73.775** (step 1: 74.268,
  spread 0.493)
- attributed to handlers: **68.978** over **21936** calls — the rows sum to this exactly
- unattributed residual: **+4.797 s (+6.5% of `backward_secs`)** — `Tape.backward`'s own loop
  and bookkeeping, **net of** the per-handler sync this probe adds. The sign is the reason it
  is stated as a residual and not as overhead: the probe's syncs push `backward_secs` up while
  unhooked loop time pushes attribution down, and only the net of the two is observable here.

**Per-call is where GDN's size comes from.** 118.0 ms/call against full attention's 4.414 —
**26.7x** — with 3x the calls (384 = 48 GDN layers x 8 rows, against 128 = 16 attention
layers x 8).

Artifact `/work/bwdops.json` on the pod, tree stamp `4bad082`.

## Controls

| control | reading |
|---|---|
| the handler is drained inside the timed span | a generator handler is inert until consumed; an `iter(fn(...))` wrapper leaves every row at **0.0 s** while the work happens in the caller. Selfcheck times a drained handler at 0.025 s against 0.021 **ms** for the undrained call |
| the timer is exclusive, not inclusive | mutating it to inclusive fails the nesting arm at `outer is charged for inner`: 0.055 s against its own 0.026 s |
| the shares are disjoint | they sum to 100.0% on tiny, and `checkpoint` drops 24.20 → 2.10 ms once nesting is subtracted |
| `main()`'s imports resolve | nine `(module, name)` pairs asserted. Four invented paths (`tilerl.lora`, `.optim`, `.sampling`) survived every local run because `--selfcheck` returns before reaching them, and died **40 s into a pod run** with `ModuleNotFoundError`. Control: pointing one back at `tilerl.lora` reproduces it locally in a second |
| the tree the numbers came from | stamp and both file shas read in the same call as the table |
| the rest of the box | all 8 cards' util and memory before and after. Unlike the earlier rows today, the neighbouring job had finished: **8/8 cards at 0 MiB, 0%** before the run |

## Not established

- **The absolute seconds are not the shipped path's.** A device sync around every handler is
  overhead training does not pay, reported as its own figure. The **shares** are the quotable
  part.
- **Not a kernel profile.** A row includes python, dispatch and allocation, so a large line
  says "look here", not "this kernel is slow".
- **The step time is not comparable to #192's 133 s.** This probe builds the engine
  `decode_graph=False` (`prof_backward_ops.py:237`) where `prof_grpo_step.py:230` uses
  `True`, so its rollout is eager and slower. That is the on-policy rule, not an oversight —
  `grpo_loop` refuses a training engine with the decode graph on. The backward is unaffected:
  the graph covers decode ticks, and nothing in `tape.backward` is captured.
- One warm step, one process, one shape.
- **The two big rows are not verified down to the kernel.** `linear_attn_chunk`'s 45.3 s is a
  handler total; which of the chunkwise-WY calls inside it dominates is unmeasured, and the
  next step for that lever is a second profile inside the handler, not a rewrite.

## Rule

Split the bucket before choosing the lever. Two candidates had been argued for two days on
plausibility — the recomputed MLP forwards and a TileLang attention backward — and the profile
puts them at 3.6% and 0.8%. One afternoon of probe time refuted both and named ops nobody had
mentioned.

A shared function is not a shared kernel: `linear_fp4_frozen` and `linear_fp8_frozen` both call
`linear_frozen_bwd`, and only one reaches a TileLang kernel. Resolve the guard, not the call.

A negative overhead is a contradiction, not a small number. The probe printed
`sync overhead: -4.797 s`; the sign was the only thing that showed the handlers do not account
for all of `backward_secs`.

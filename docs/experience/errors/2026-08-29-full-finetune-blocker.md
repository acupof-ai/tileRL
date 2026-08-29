# Full fine-tuning: not "does not fit" but "never wired", then OOM in the backward — 2026-08-29

> Status: measured. Both claims that preceded this entry were wrong.

## What the tree said

`bench_harness.suite_train`:

> Full-parameter 27B needs 54 GB of bf16 masters + 216 GB of Adam moments and
> does not fit one H20 at any shape.

I quoted that and then extended it, computing 270 GB resident and adding a
third term (AdamW's per-parameter `p32 = p.to(float32)` temporary). None of it
had been measured, including by me.

## What is actually true

**1. Without `keep_master`, "full-parameter" trains 402 tensors, not 27B of
them.** The fp4 base is frozen and its backward yields dX only, so the tape
returns gradients for the norms, `conv1d`, `dt_bias` and `a_log` — **4.89 GiB**
in total, and the step runs fine. It was never a memory problem; the weights
simply had no gradient.

| 27B, T=256 | LoRA | "full" (no master) |
|---|---:|---:|
| tape entries | 2678 | 1187 |
| returned param grads | 1396 / 5.34 GiB | 402 / **4.89 GiB** |
| backward peak | 56.6 GiB | 48.5 GiB |
| **held past the result** | **8.88 GiB** | **9.32 GiB** |

**2. With `keep_master=True` it OOMs — in the backward, not at load.** 95.09 of
95.22 GiB, failing inside `gemm_tn` allocating a weight gradient. The masters
(54 GB) load fine. What does not fit is **every weight gradient existing at
once**, which is what `Tape.backward` requires: it accumulates into one dict and
returns only after the last entry.

So the ordering is the opposite of what I assumed. Optimizer state is not the
blocker; it is never reached.

**3. `master_linear` is not an STE.** With a master present the forward switches
to the dense bf16 `linear` — 54 GB of weights instead of 13.5, and the fp4
kernel's bandwidth advantage gone. It is bf16 training, not fp4-forward with a
bf16 master.

## Two fixes, in this order

- **Consume gradients during the reverse pass.** Apply the optimizer to a
  parameter as soon as its gradient is final and drop it, instead of collecting
  all of them. This is what makes full fine-tuning possible at all, and it also
  reclaims the **~9 GiB** of intermediate-activation gradients the dict holds
  past their last use — which LoRA pays too (16% of its peak).
- **Then** the optimizer state, where Adafactor's factored second moment
  replaces Adam's `m`+`v`.

## Rule

A "does not fit" in a comment is a hypothesis with a number attached, not a
measurement. This one was wrong twice over: the configuration it described was
not the one anybody was running, and the real failure is in a different
component than its arithmetic named. Run it before extending it — I spent two
rounds refining an estimate of a path that was never taking gradients.

# Streaming gradient release freed the tail, not the peak — REVERTED

## Context

Goal: make full fine-tuning of the 27B fit on one card, and cut the LoRA step's
memory. `scripts/probe_tape_mem.py` measured that after `Tape.backward` returned,
the gradient dict still held **8.88 GiB** on the LoRA path (16% of its 56.6 GiB
backward peak) and **9.32 GiB** on the full path — gradients of frozen params and
of the GDN initial state, none of which the step uses.

So `Tape.backward` gained an `on_grad` callback: a gradient is offered the moment
it is final and dropped unless the caller keeps it. `train._step` kept only
parameter gradients. Numerically identical — there is an equivalence test
(`test_backward_streaming_matches_collecting`) and `clip_grad_norm` still sees
every parameter gradient.

## Root Cause

**The 9 GiB was a post-return figure; the peak happens mid-backward.** At the
moment of peak, those intermediate gradients are still live and still needed —
dropping them earlier shortens the tail after the peak has already passed.
Reclaiming memory the allocator was about to reclaim anyway buys nothing.

Measured, `bench_harness --suite train`, 27B LoRA, GPU 7:

| B x T | tok/s before | tok/s after | peak GB before | after |
|---|---:|---:|---:|---:|
| 1x64 | 50.3 | 52.3 | 47.0 | 47.0 |
| 1x128 | 80.5 | **75.4 (0.937x)** | 50.6 | 50.5 |
| 1x256 | 113.5 | **105.9 (0.934x)** | 57.5 | 57.6 |
| 2x256 | 178.2 | **172.7 (0.969x)** | 76.5 | 71.8 |

One row of the four saves anything (2x256, -4.7 GB). Three regress 3-7%: the
release pass costs a `_first_use` scan of the whole tape plus a `list(grads.items())`
per entry.

## Fix

Reverted the wiring in `train._step` back to `grads = tape.backward(grad_logits)`
plus the id filter. `Tape.backward(..., on_grad=)` is kept — it is correct, gated
behind a default of `None`, has its equivalence test, and is the mechanism a real
fix needs (consume-and-discard each parameter gradient inside backward, which is
what would let full fine-tuning fit). It is simply not, by itself, a memory win.

## Rule

A "held after the call returns" number does not predict peak memory. Peak is a
property of the busiest instant, and anything released after that instant is
free either way. Before optimizing a memory figure, check that the figure is
measured at the peak — otherwise the same class of error as pricing a kernel
in one configuration and spending it in another
([wins/2026-08-29-spec-decode-net-win.md](../wins/2026-08-29-spec-decode-net-win.md)).

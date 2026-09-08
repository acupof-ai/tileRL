# 27.2 s of "sampling" was the forward draining at the first host read — 2026-09-08

**Status:** fixed (`scripts/probe_rollout_breakdown.py`, the `--sync-forward` arm and the
weight-stream floor assert).

## Context

`timings["rollout_secs"]` is 63.156 s of an 85.617 s GRPO step and nobody had looked
inside it. A probe wrapped `engine.step()`'s own methods and reported, on card 6:

```
step 1:   30.418 s   8192 tok   269.31 tok/s
    _run_forward     30.382 s   99.9%
    _sample_commit   27.235 s   89.5%
```

Read as "89.5% of the rollout is host-side sampling", extrapolating to 56.5 s of the
GRPO step — 66% of it — in a logits→token path nobody knew existed.

It was the forward. Sampling is 1.8 ms/tick.

## Root cause

CUDA launches are asynchronous. There is **no synchronization anywhere on the decode
path** between `self._model.forward` (`engine.py:1025`) and `toks.tolist()`
(`engine.py:1347`) — enumerated across `engine.py`, `model.py`, `backend.py` and
`reference.py`; the only other sync, `b.synchronize()` at `engine.py:1294`, is on the
speculative path and this run did not take it.

So `_run_forward`'s timer stops when the last kernel is **queued**, and the queue drains
at the first host read, which is `.tolist()` inside `_sample_commit`. Wall clock on an
async device attributes to the first blocking call, not to the work.

## The control

`--sync-forward` adds one `torch.cuda.synchronize()` after `_model.forward` and changes
nothing else. The two arms are mutually exclusive by construction, and the measurement
picked one:

| ms/tick | async arm | sync arm | microbench |
|---|---:|---:|---:|
| forward | 3.07 | **89.50** | — |
| sampling | 26.60 | **1.81** | 1.065 |

The microbench is the independent path: `torch.sort` over the full vocabulary, `topk`,
and the per-row `Generator`+`multinomial` loop, timed on card 7 in their own process
against a peaked fixture. It says the sampler's real operators total 1.065 ms/tick, which
no reading of the async arm's 26.60 ms can accommodate.

## What made it visible before the control ran

The **weight-stream floor**, computed from the first arm's own numbers. Exclusive forward
was 3.07 ms/tick; `qwen38-27b` is dense (`config.py:119-140` — no expert fields), so a
decode tick reads every parameter, and at H20's 4 TB/s nameplate that takes **6.11 ms**.
A forward cannot finish in half the time needed to read the weights it multiplies by.

Two resident readings exist and the floor does not turn on which: 22.76 GiB (24.44 GB,
read on the live card 2026-09-08) gives 6.11 ms, and 23.22 GiB (24.93 GB,
`wins/2026-09-03-grpo-27b-fits-the-card.md:49`) gives 6.23 ms. The probe uses the lower,
which is the conservative choice for this argument. 3.07 ms violates both.

The floor's two assumptions, since it was load-bearing before the control existed: the
nameplate bandwidth (real achieved rates run 0.7–0.9 of it, which would raise the floor
and strengthen the violation) and one read per parameter per tick (cache hits or fusion
would lower it). So it licenses "3.07 ms violates a nameplate, one-read-per-parameter
bound", not "3.07 ms is physically impossible". It was enough to justify spending a card
on the control; the control is what settled it.

The probe now asserts this on the sync arm. It is the check the reconstruction gate could
never be: the async arm's table was perfectly self-consistent — parent 99.9% of wall,
child 89.5% of parent, remainder 0.0% — and every one of those percentages was true. A
decomposition can be internally exact and still attribute to the wrong phase, so the gate
that catches it has to come from outside the decomposition. Physics is outside it.

## Three defects in one instrument

1. **Flat partition** — `_sample_commit` and `_finish_prefills` nest inside
   `_run_forward`, so summing four inclusive timers double-counted. Caught by me.
2. **An assert that could not fail** — `acc["unattributed"] = wall - sum(acc.values())`
   defined the remainder, then asserted the parts summed to `wall`. It printed
   `unattributed = -27.210 s` and passed. A negative remainder is the loudest available
   alarm, and the gate turned it into a row.
3. **Async attribution** — this entry.

The first two are arithmetic and were fixed from the desk. The third needed a card, and
it survived both earlier fixes: correcting the nesting made the table *more* self-
consistent while leaving the attribution wrong.

## Cost of the control

The sync arm is not a free observation. `torch.cuda.synchronize()` inside a captured
region raises `cudaErrorStreamCaptureInvalidated`, the decode graph capture is abandoned,
and the run silently falls back to eager — 93.16 vs 29.71 ms/tick, **3.14x**. The arm
still answers the attribution question, because attribution is a question about which
timer holds the time, not about how much time there is. But its absolute numbers are
eager numbers and cannot be quoted against the captured path. The probe now guards the
sync with `torch.cuda.is_current_stream_capturing()`.

## What the number actually is

Captured, async arm, 29.71 ms/tick total: sampling 1.065 ms (3.6%), everything else
28.64 ms (96.4%). Against the 6.11 ms floor the decode forward runs at **21.3% of the
HBM bandwidth bound**. That is where the rollout time is, and the question moves inside
the kernel — occupancy, KV traffic, the GDN state — not to the sampler and not to spec
decode.

The two sampler-side opportunities that are real: the per-row `Generator`+`multinomial`
loop is 9.35x a batched draw (0.790 → 0.085 ms) and `topk` beats the full-vocabulary sort
by 2.26x (0.274 → 0.121 ms). Together **3.2% of the rollout**, not 66%.

## Rule

**A timer on an asynchronous device measures where the queue drains, not where the work
happens.** Before decomposing a GPU path, enumerate its synchronization points: the phase
holding the first host read is billed for everything queued ahead of it. If there is no
sync between two phases, their split is an artifact.

**A self-consistent decomposition is not a correct one.** Reconstruction gates check the
arithmetic of the partition, which is exactly the property that survives misattribution.
The check must come from outside — a hardware floor, an independent microbench, or a
control arm that moves the boundary being tested.

**Price the control's side effects before reading it.** A synchronize changes the
execution mode, not only the observation, and an arm that answers the attribution question
can be worthless for absolute numbers. Say which of the two you are quoting.

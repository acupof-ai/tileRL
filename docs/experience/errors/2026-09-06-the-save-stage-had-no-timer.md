# The one spill stage with no timer was the one with a number — 2026-09-06

> Status: partly open. The write side is measured here on a 499.6 MiB/s host SSD;
> the pod's device is not measured, so the pod figure is an extrapolation and is
> labelled as one. Listed in OPEN.md.

## Context

`~100 ms` for the tier's `torch.save` appears in five places — `kv_cache.py:433`
and `:861`, `bench_write_through.py:4`, and twice in
[the SSD tier's read path did not exist](../wins/2026-09-05-the-ssd-tiers-read-path-did-not-exist.md).
It is load-bearing: it is the stated reason the save is handed to a daemon
thread instead of running on the publish path.

## Root Cause

**The spill has six stage timers and none of them wrap the save.** `gather_ms`
and `copy_ms` at `kv_cache.py:584` and `:617`, `demote_ms` / `promote_ms` on the
DRAM tier — and `torch.save` at `:511` had nothing. The comment above those
timers says why they exist: "three guesses at the per-publish cost were wrong in
a row, so the spill reports where its time goes instead of being guessed at a
fourth time." The instrument was built to stop exactly this, and skipped the
largest stage in the function.

**Measured**, `scripts/probe_save_ms.py`, driving `spill_state` directly (no GPU,
no model — the save is a daemon write of a CPU blob), distinct contents per entry
so nothing is served from a repeat:

| entry size | saves | ms/save | implied MiB/s |
|---:|---:|---:|---:|
| 32.0 MiB | 3 | **54.7** | 585.4 |
| 320.6 MiB (one real entry) | 4 | **641.8** | 499.6 |

**641.8 ms against the asserted ~100 — 6.4x**, on this Mac. Cost is
byte-dominated rather than fixed-overhead: 10.02x the bytes costs 11.74x the
time, so extrapolating by bandwidth is sound in form even though the device here
is not the pod's.

## Two of my own numbers were wrong first, in the same direction

Before running anything I derived the discrepancy twice and published neither:

*8.6x*, from 156.9 MB — the GDN state only, dropping the 167.8 MB of KV that the
same entry carries.

*17.6x*, from 320.6 MiB over **182.6 MiB/s** — and that rate is a **read**.
[the SSD benchmark never touched the SSD](2026-09-05-the-ssd-benchmark-never-touched-the-ssd.md)
measured it on the two files **one fault-in reads**, three ways. Substituting a
read rate into a write path has no basis: they are different code paths on the
device and nothing in the tree measured the write side until this probe. The
arithmetic was clean, the operand was the wrong quantity, and the error made my
own finding look larger — which is why it went unchecked through two derivations.

So the honest form of the finding is: the assertion is wrong by **6.4x on a
499.6 MiB/s device**, and the pod's write rate is **unmeasured**. If the pod
writes at the 182.6 MiB/s its reads do, one save is ~1.76 s and the factor is
17.6x — but that conditional is an extrapolation across an unmeasured axis and is
not a measurement.

## What the 6.4x does and does not change

**Does not change the design.** The save being off-tick is more justified at 642
ms than at 100, not less. No runtime behaviour changes here.

**Does change one number that gates a decision.** `max_pending=32`
(`kv_cache.py:405`) bounds in-flight writes, and its comment reasons about
whether "a 229 MB/s spinning device keeps up with write-through". At 642
ms/save a full queue is **20.5 s** of backlog holding 32 × 320.6 MiB = 10.0 GiB
of host RAM, on a host the same comment calls 31 GB. The cap was set against a
per-save cost 6.4x too low — the same shape as
[the eval cap measured itself](2026-09-04-the-eval-cap-measured-itself.md) and
[the rollouts grew into the cap](2026-09-06-the-rollouts-grew-into-the-cap.md):
a bound whose operand was never measured. **Not fixed here** — sizing it needs
the pod's write rate.

## A second defect I reported to myself and then withdrew

While reading the cap I found that `spill_kv` checks
`len(self._pending) >= max_pending` and counts a refusal, while `spill_state`
(`:655`) has **no capacity check, no `_healthy` check, and returns `None`** so it
cannot refuse. A probe driving `spill_state` directly, with a KV-path control arm,
read a **peak of 32 blobs against a cap of 4 with 0 refusals** — 4.90 GiB at a
real 156.9 MB snapshot. The control arm was red for the right shape (KV peaked at
exactly 4 with 28 refusals), so the reading looked solid.

**It measures a call sequence no caller performs.** `PrefixStore.insert` at
`:875-877` reaches `spill_state` only inside `and self._ssd.spill_kv(...)` — the
KV refusal gates both halves, by design, because "both halves go or neither — a
fault-in needs the pair." Re-run through the real pair: **accepted 5, refusals
27, `_pending` peak 4, `_pending_st` peak 5**, stable across three runs. The +1 is
not an over-cap window: the two tables drain on different clocks, and a 64 MiB
state blob finishes its save after the small KV blob that gated it. No defect.

My control arm was the other code path, not the same path with its guard removed,
so it could not distinguish "no check here" from "checked upstream". A guard's
absence at one call site is not a missing bound; the bound can live in the only
sequence that reaches it.

## Fix

`ssd_save_ms` and `ssd_saves` are published together in `stats()`. Both, because
the claim is a **per-save** cost: a total alone cannot be divided by a count the
caller never sees, and reporting a total against a per-save assertion is how a
mean gets fabricated. `scripts/probe_save_ms.py` prints `sorted(st)` — every key,
not a hand-listed subset, after a probe three rounds ago reported a partition of
the data instead of the data.

Not run on the pod: the tier's directory is `/work`, which is inside the
container namespace and invisible to a plain `ssh v100`, and the write probe
moves 1.25 GiB through the same device the live endpoint is serving from — a
write-bandwidth measurement is exactly what contention corrupts.

## Rule

**A stage with a number and no timer is the one to instrument, and a set of
timers is not evidence the expensive stage is among them.** Five sites agreeing
looked like support; the function's own timers looked like diligence. Check that
the instrument wraps the step the claim is about.

Second: **a rate has a direction.** Read and write are different paths on the
same device, so a measured read bandwidth is not an operand in a write
calculation. My two pre-run derivations both used the wrong operand and both
overstated the defect, and an error that makes your own finding bigger is the
one you do not re-derive.

Third, from the withdrawn second finding: **a probe that calls the function
directly is not measuring the system unless a caller calls it that way.** Both
probes here produced state rather than reading code, and one of them still
produced a number about nothing — a 32-vs-4 reading on an entry point whose only
caller reaches it behind another function's refusal. Produce the state *the
callers produce*: find the call site first, then drive that.

## Results

Runtime change: two counters added to `KvTier.stats()`, no behaviour change.

| date | commit | machine | measurement | value |
|---|---|---|---|---|
| 2026-09-06 | (this) | Mac host SSD | `torch.save`, 320.6 MiB entry | **641.8 ms/save**, 499.6 MiB/s |
| 2026-09-06 | (this) | Mac host SSD | `torch.save`, 32.0 MiB entry | 54.7 ms/save, 585.4 MiB/s |
| 2026-09-06 | (this) | — | the asserted figure, 5 sites | ~100 ms — **6.4x low here** |
| 2026-09-06 | (this) | pod `/work` | write rate | **unmeasured** — namespace + live endpoint |

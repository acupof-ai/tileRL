# Optimizing at the wrong layer: scheduling a KV move that should happen once — 2026-09-21

> Status: **rule, not a defect.** This entry records a direction correction (Epic
> #779, M7). Its subject is the close/transfer optimization layer that #741 / #743
> / #745 / #746 / the busy-idle probes built, all of it **default OFF** and none of
> it on the production path. The refactor removes it and publishes a page once, when
> it leaves the pool. Before numbers below are vendored and recomputable; no new
> measurement was taken for this entry.

## Context

The training workload issues **structurally disjoint spans**: consecutive requests
share no prefix, so the content-addressed publish path is entered and the shared
lookup never hits. Against that workload a 32k request still paid, once per close:

- **09-21 final V100 window, n=20 close brackets** (all closes, not a tail —
  `errors/kv-once-before-close-2026-09-21/close-ticks.tsv`, recomputable):
  `release_close_request` median **1829.5 → 1098.5 ms** (baseline → `TILERL_CLOSE_BATCH_D2H=1`,
  −40%); components at the same n=20 median `ssd_mmap` 716.5 → 660.0 ms and
  `pub_cold_transfer` 389.5 → 363.0 ms.
- **09-19 window, ≥1 s CLOSETAIL ticks only** (a different geometry, n=6/7 —
  `wins/close-batch-cap-device-2026-09-19/close-segments.txt`): `ssd_mmap` med
  1833 ms and `pub_cold_transfer` med 1726 ms on cb0, i.e. **~1.8 s + ~1.8 s inside
  one close** — with **zero readers** for the published bytes on the train span.

The two geometries are both real and must not be quoted interchangeably: the 09-21
figure is the median of every close; the 09-19 figure is the median of the slow
tail only.

## Root Cause

The optimization effort went into **scheduling** a one-shot transfer, because the
workload was read as "KV movement that happens every step and needs to be spread
out". The cost model that follows from that reading is wrong in a specific way:

- What attention actually loads comes from HBM, at most once per page per
  residency period. That is the "load" that would be worth engineering.
- What was optimized instead is the **close-time host/SSD→shared transfer**, which
  is by construction a **one-shot per page per lifetime** — it happens when the page
  is finished, not every step.
- A dispatcher was then built for it: #741 batch D2H, #743 a background thread with
  its own future/adoption protocol, #745 a derived queue depth plus a payload byte
  cap, #746 the disk passes moved off the tier lock, and instrumentation for all of
  it. Each layer is defensible in isolation and every one of them exists to make a
  transfer the workload did not need survive a critical path it should not have been
  on.

Every piece was **default OFF**, and the production path ran inline — so removing
the layer cannot change production behaviour.

## Fix

Publish **once, at the moment the page finishes and leaves the pool** (the existing
`offer_drop → publish_dropped → share_hold` path, content-addressed and already
deduplicated). Then close is **zero bytes**: `_release` no longer forces a close and
no longer D2Hs hot pages. A prompt whose pages stay resident in device memory is
simply not shared — accepted, and immaterial to a train workload of disjoint spans.

Everything listed above is deleted: the batch D2H context, the publisher thread and
its future protocol, the depth/payload sizing, and the
busy/idle + worker-mmap instrumentation.

**One thing is retained, and it is the trap in this entry.** #746's lock-split primitives (`ColdSsdFile._mlock`, `HostKvPages.share_hold_kv`) are **not** deleted: the background worker that held them is gone, but the natural-leave path calls `share_hold_kv` inline (`offer_drop → transfer_to_shared → share_hold_kv`), on the step thread, once per page. What was removed is the *calling form* — a worker doing disk IO with the lock released — not the primitives. A reader who takes "the lift was deleted" literally will delete live code.

## Rule

**Ask how many times an operation happens before asking how to schedule it.** A
scheduler built for an operation that should occur once is the most expensive shape
available: it costs the dispatcher itself, its queue and locking, its tuning
constants, and the measurement arms needed to watch it — and its entire payoff is
confined to making that should-not-happen operation cheaper. The tell is a
default-OFF gate whose enabling condition is a workload property (here: prefix
reuse) that the workload in front of it does not have.

## Provenance

- `errors/kv-once-before-close-2026-09-21/close-ticks.tsv` — the 40 raw close rows
  behind the n=20 medians above (2 arms × 20 closes × 8 component columns).
- `wins/close-batch-cap-device-2026-09-19/close-segments.txt` — the 09-19 CLOSETAIL
  medians.
- `wins/bg-publish-device-2026-09-20/` — the #743 arm artifacts.

**Convention and a collision note.** The medians above are taken over **all 20 close
ticks per arm** (one warmup close is 0), so an even n averages the middle pair:
1829.5 and 1098.5. The nonzero-only medians are 1831 and **1104** — quoting
1831→1098 would mix the two conventions. Separately, 1831 and 1098 (and 1104) also
appear as ordinary single-tick `forward=` / `total=` millisecond values in unrelated
V100 serve logs. They are not the close medians. Read the vendored column, do not
grep the number.

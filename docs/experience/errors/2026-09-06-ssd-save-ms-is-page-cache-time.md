# `ssd_save_ms` is page-cache time on the host that matters, and my Mac inverted every conclusion — 2026-09-06

**Date:** 2026-09-06
**Hosts:** this Mac (APFS, NVMe) **and** the H20 pod's `/work` (`/dev/vda2`, 2.0T, 84% full)
**Commit:** 7f07abd
**Verdict:** the 641.8 ms cited in five places and used to re-size `max_pending` is the
**page-cache** cost. On the pod the durable cost of the same entry is **1337-1389 ms**, 2.1x
higher, and `max_pending=32` was being sized against the low number.

## Context

`kv_cache.py:518` is a bare `torch.save(blob, dst)` with the timer wrapped around exactly that
call and **no `fsync`**. A buffered write returns when the kernel accepts the bytes, not when
the device has them, so `ssd_save_ms` is a lower bound on the real cost. The
[save stage had no timer](2026-09-06-the-save-stage-had-no-timer.md) entry corrected five
`~100 ms` comments to a measured 641.8 and left `max_pending` open in OPEN.md; that figure is
what the cap is now being sized against, so which quantity it is decides the cap.

## What was measured

`scripts/probe_save_fsync.py`, five arms on the same 320.6 MiB entry (156.9 MB GDN state +
167.8 MB KV), medians of 5 reps, distinct contents per rep so no arm can be served a repeat:

| arm | Mac ms | pod `/work` ms |
|---|---:|---:|
| `save` (what the tier times today) | 535 | **273** |
| `save` + `fsync` (durable) | 537 | **1337** |
| raw `tobytes()` + write + fsync | 42 | 1708 |
| `torch.save` to a `BytesIO` (no disk) | 71 | 288 |
| serialize to RAM, then one write + fsync | 122 | 1430 |
| volume sequential rate, median of 3 | 4865 MiB/s | **185 MiB/s** |

**fsync multiplies the timed cost by 0.98x on the Mac and 5.75x on the pod.** Reproduced on a
second pod run (273 / 1337 against 241 / 1389).

## Every conclusion I drew from the Mac was wrong for the pod

Recorded in this order because the order is the finding:

1. **"fsync adds nothing, so 641.8 is a real per-save cost."** True on APFS at 0.98x. On the
   pod it is 5.75x, so the tier's timer stops five sixths of the way before the bytes are
   durable. The claim survived a control and was still local-only.
2. **"torch.save is 11.6x a raw write, so the tier is not disk-bound."** The Mac's raw arm was
   42 ms against 537. On the pod the raw arm is the **slowest** of the five (1708 ms) and
   `save+fsync` is **0.81x** it — the pod IS disk-bound, at 187.6 MiB/s, 1.02x its own volume
   rate.
3. **"Serialize to RAM then write once: 122 vs 537 ms, a 5.24x win, same bytes, no dtype
   change."** This was the moment it looked like a shipping optimisation. On the pod it is
   1430 against 1337 — **7% slower than what the tier already does.** The win was APFS
   coalescing one large write better than torch's incremental writer; the pod's device has no
   such headroom.

The Mac reads **26x** the pod's volume rate (4865 vs 185 MiB/s). At that ratio nothing about
which stage dominates transfers, and I would have opened a PR changing the save path on the
strength of arm 3.

## What the number means for the cap

`max_pending=32` holds 32 × 320.6 MiB = **10.0 GiB** of a host with 1928 GiB total and 1473
available (measured, `free -g`), so the OOM framing in the `kv_cache.py:401` comment is not the
live risk — the cap's real job is bounding how far the queue can run ahead of the drain.

Drain rate on the pod, from the durable figure: **1337 ms per 320.6 MiB entry = 240 MiB/s**.
A full queue of 32 therefore takes **42.8 s** to drain, against 20.5 s if 641.8 were the cost.
Sizing the cap on 641.8 understates the drain time by 2.1x — the direction that makes the cap
look safe.

**Not proposing a number here.** The cap should be sized against a measured publish rate, and
this probe measures the drain side only; a cap set against a drain rate with no arrival rate is
the same defect as [the rollouts grew into the cap](2026-09-06-the-rollouts-grew-into-the-cap.md),
one axis over — a bound set without measuring the thing it bounds. The OPEN.md row stays, with
the operand corrected from 641.8 to 1337 and the missing half named.

## Also: the "byte-identical" claim I nearly made

Before benchmarking arm 5 I checked that serialize-to-RAM-then-write produces the same file.
It does **not**: 4,196,133 bytes against 4,196,125, an 8-byte difference in the zip container.
`torch.load` returns equal tensors and equal python objects from both, which is the property
that matters — but "byte-identical, so it's a safe swap" was the sentence I was about to write,
and it was false.

## Rule

**A storage measurement does not transfer between hosts, and the direction of the error is not
predictable.** Three conclusions, all correctly measured, all controlled, all inverted by the
pod: fsync-free (0.98x → 5.75x), not-disk-bound (11.6x → 0.81x), and a 5.24x optimisation that
is a 7% regression. The Mac is 26x the pod's write rate; the ratio should have been the first
thing measured, and it would have stopped arms 2 and 3 from being written up at all.

**A timer with no `fsync` measures the kernel, not the device.** Any per-write cost quoted from
a buffered call needs the durable arm beside it, and on the host the cap actually runs on.

**Run the arm that would change the code on the target first.** Arm 3 was the only arm with a
patch attached, and it was the one that reversed.

## Results

| date | commit | host | fs | shape | metric | value |
|---|---|---|---|---|---|---|
| 2026-09-06 | 7f07abd | H20 pod | `/work` vda2 | 320.6 MiB entry | `torch.save`, as timed today | **273 ms** |
| 2026-09-06 | 7f07abd | H20 pod | `/work` vda2 | 320.6 MiB entry | durable (save + fsync) | **1337 ms** |
| 2026-09-06 | 7f07abd | H20 pod | `/work` vda2 | 320.6 MiB entry | durable drain rate | **240 MiB/s** |
| 2026-09-06 | 7f07abd | H20 pod | `/work` vda2 | sequential 1 MiB loop | volume write rate | 185 MiB/s |
| 2026-09-06 | 7f07abd | this Mac | APFS | 320.6 MiB entry | durable (save + fsync) | 537 ms |
| 2026-09-06 | 7f07abd | this Mac | APFS | sequential 1 MiB loop | volume write rate | 4865 MiB/s |

No runtime change — one probe and this entry. The cap is untouched, with its operand corrected.

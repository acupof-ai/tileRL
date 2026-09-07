# Row 50 — SSD read path: approach

From `origin/main` at `d4fc973`. No code. Numbers marked **measured** are read off
an entry; the rest are derived and named as such.

## The read path already exists

`PrefixStore.lookup` → `_fault_in` → `KvTier.load_kv` / `load_state`, landed
2026-09-05 (`wins/2026-09-05-the-ssd-tiers-read-path-did-not-exist.md`), measured
**1.738–1.821x** on a 2729-token first turn after a restart. So row 50 is not
"build a read path". It is two additions to one that works and never fires:

1. it faults in unconditionally, at whatever length happens to be resident;
2. it faults in **synchronously, inside `_admit`, under `Engine._lock`**.

`ssd_hits` was **0 in all 132 rows** of the 12-session bench
(`errors/2026-09-06-the-ssd-tier-is-165x-worse-at-12-sessions.md`) — that arm never
restarted, so the pool always answered first. The 1.65x there is pure write cost.
Neither addition below changes the write path.

## 1. Break-even length, derived at start

Fetch beats recompute when

```
(S + n·k) / B  <  n / R
n*  =  (S/B) / (1/R − k/B)          # tokens; if k/B ≥ 1/R, fetch never pays
```

| operand | what it is | where it comes from at runtime |
|---|---|---|
| `k` | KV bytes/token | `2 · len(full_attn_layers) · num_kv_heads · head_dim · itemsize` off `PagedKvPool`. **Per arch, and it differs by 2x**: the sm70 pool is f32 (`backend.py:353` sets `io`, `engine.py:1527` takes it as the pool dtype), so 16 full-attn layers × 4 kv heads × 256 gives **64.0 KiB on the H20** (bf16) and **128.0 KiB on the V100** (f32) |
| `S` | recurrent snapshot, fixed per prefix | one `spill_state` blob size, or `snap.numel()·itemsize` at build. **Measured 149.63 MiB**, 4.6 KB spread across four lengths |
| `B` | tier read B/s | **the cumulative rate over every fetch so far**, not one calibration read: `_fetch_loop` accumulates `fetch_ms` and `fetch_bytes` on each load -- both planes -- and `read_bytes_per_s` divides the running totals. **Measured cold 182.6 MiB/s** (H20 host, 09-05 tier bench) and **191 MB/s = 182.2 MiB/s** (V100, virtio-blk, three reps to 0.07%, block layer confirmed) — the two disks are the same speed to within 0.2%. Cold is the rate this table's `n*` uses; the page cache reads **28x faster** (measured 5.664 GB/s warm against 0.203 cold on /data00), so see the warm-start note below |
| `R` | prefill tok/s, this arch | **Not a scalar on the V100.** sm70 prefill is quadratic in `n` (#213's TTFT fit, `0.56 + 0.00422n + 4.117e-7n²`), so `n/R` becomes that fit: 5.3 ms/token at 2k, **16.6 ms/token at 30k**. On the H20, **measured 2558.6 tok/s** (`prefill/len8192/sm90`, seeded baseline, contended box, top of range) and flat *within the seeded range only* — the note does not extrapolate it to 30k. This operand is what separates the two archs |
| `n/R` | the fetch deadline | not a fourth constant: the same `n`, `R` as above. A fetch still in flight at `n/R` from issue is **dropped** and the request prefills. It held no blocks, so the drop frees nothing and races nothing; the reader thread's buffer is discarded when it lands, and `insert` is never called |

`S` is a fixed 149.63 MiB whatever the length, so at short prefixes the tier is
paying to read a snapshot it barely amortizes; `n*` is where the KV it also reads
finally covers that cost.

| arch | `k` | recompute | `k/B` | `n*` | 30k fetch | 30k recompute | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| **H20** | 64 KiB | 0.391 ms/tok flat | 0.342 ms/tok | **~16,900** | 11.1 s | 11.7 s | **1.06x** |
| **V100** | 128 KiB | #213 fit, 16.6 ms/tok at 30k | 0.686 ms/tok | **~73** | 21.4 s | 497.7 s | **23.3x** |

The V100 row uses the quadratic fit rather than a flat rate, which is what makes its
ratio grow with length: 4.9x at 2k, 9.7x at 8k, 14.9x at 16k, 23.3x at 30k. A flat
75 tok/s would have read 18.7x at 30k and understated every shorter length.

**The two archs are different problems, and the H20 is the hard one.** Its prefill is
an order of magnitude faster while its disk is the same speed, so its recompute and
read rates are within 14% of each other: 0.391 ms/token recomputing against
0.342 ms/token reading. `n*` lands near 17k tokens and even a 30k prompt is only
**1.06x** on wall clock. The V100 reads at 0.686 ms/token against a recompute that
starts at 5.3 ms/token and *worsens* with length, so it wins from ~73 tokens up and
by more the longer the prompt.

Sanity on the formula, against the one real fault-in: at n=2560 with the 09-05
operands it predicts a 309.6 MiB entry against **320.6 MiB measured** (1.035x) and
1.70 s against **1.756 s measured**. It is 3.5% light on bytes, so no `n*` here is
worth more than two significant figures.

### A warm first fetch over-permits, for one or two fetches

`B` is measured from this tier's own reads, so whatever the page cache holds at the
first fetch sets it. On a restart into a warm cache that first read is memory-speed —
5.664 GB/s measured against 0.203 cold, **28x** — and `n*` collapses with it: at a
2,700-token entry (337.5 MiB of KV on the V100) it reads **6 tokens** instead of the
cold **195**, so every prefix is permitted for as long as the estimate stands.

It does not stand, because `B` is cumulative rather than a calibration. One cold fetch
drags it from 5.664 to 0.392 GB/s and `n*` from 6 to 93; by the twentieth fetch
`n* = 184`, within 6% of the pure-cold 195:

| fetches (1 warm, rest cold) | `B` (GB/s) | `n*` |
|---:|---:|---:|
| 1 | 5.664 | 6 |
| 2 | 0.392 | 93 |
| 5 | 0.251 | 152 |
| 20 | 0.213 | 184 |

So the exposure is the **first fetch or two after a restart**, and the `n/R` deadline
already bounds what one costs: an over-permitted fetch that cannot finish inside the
prefill it replaces is dropped at `engine.py:714`, holding no blocks. The cost of the
whole window is a reader thread and a host buffer, twice. Not worth replacing the
measurement with a block-layer read — `read_bytes_per_s()`'s docstring says the rate
must come from this tier's own fetches precisely so it does not describe whichever box
it was written on, and a `/sys/block` probe reintroduces exactly that.

### The degenerate case is not reachable

`k/B ≥ 1/R` means the device cannot stream KV as fast as the card recomputes it.
Rearranged it is a floor on bandwidth, `B_min = k·R`: **159.9 MiB/s** on the H20 and
**7.7 MiB/s** on the V100 at 30k (higher at short lengths, where its recompute is
faster), against 182 MiB/s measured on both. So the tier pays on
both archs — but the H20 clears its floor by only 14%, the same margin as above seen
from the other side, and that is why the H20's case rests on overlap rather than on
the arithmetic.

### Host RAM is the faster tier above this one

`--dram-bytes` already exists (31 GiB on the V100 box, 19 GiB of it page cache), it
sits above the disk, and **the same formula applies with `B` = the memcpy rate**. At
a pinned-H2D order of 10 GB/s, `n*` falls to tens of tokens on both archs — the DRAM
tier is unconditionally worth reading from, and the interesting threshold is only
ever the disk's. Which tier answered is a property of `resident()`, not of this
formula, so the implementation should compute `n*` per tier rather than once.

## 2. Overlap

Today `_fault_in` runs inside `_admit`, inside `step`, inside `Engine._lock`: a
1.7 s read stalls every decode row in the batch. That is the defect, and it is worse
than the fetch being slow.

- **Issue point: `submit`, not `_admit`.** `submit` today does no prefix work at
  all — the rolling hash is computed inside `PrefixStore.lookup`
  (`kv_cache.py:927-931`), and the comment at `engine.py:538` says the match was
  moved to `_admit` deliberately, because `submit` has no later tick to retry an
  allocation on. So this needs a **new, allocation-free probe** in `submit`: roll
  the hash, ask `resident()` — a dict probe, no I/O — and enqueue the fetch. That
  reuses the hash loop but not the match, and it takes no blocks, so the reason the
  match lives in `_admit` does not apply to it. By the time `_admit` runs, the bytes
  are in a pinned host buffer or still coming.
- **Who does the read.** The tier already owns a writer thread; a reader thread of
  the same shape does `torch.load` into pinned memory off-tick. Only the H2D copy
  touches the GPU, on a side stream with an event — the upgrade path the 09-05 entry
  named, still unbuilt.
- **What the queue does while it is in flight.** Nothing new: the request stays in
  the waiting queue and `_admit` returns False, which is the existing
  does-not-fit-yet path (`errors/2026-09-07-a-prompt-that-does-not-fit-yet-is-queued.md`).
  Other requests admit and prefill normally — that is the overlap. **A fetch must
  not hold blocks while it waits**, or it converts into the block-starvation case.
- **Entering the pool.** Unchanged: `_fault_in` allocates, `load_kv` fills, `insert`
  adopts. Async needs one addition — the event must be waited on before `insert`
  publishes, since a published entry can be adopted by another request the same tick.
  **No block-granular change is required.** The block-granular store is what makes
  the tier serve a *running* workload; the restart case this row targets does not
  need it.
- **Deadline.** `n/R` from issue, the operand in the table above: past it, admit and
  prefill, and discard the fetch when it lands. Without a deadline a slow read is
  unbounded and the break-even is an expectation rather than a bound.

## 3. Acceptance

Two arms, fetch vs recompute, one server per arm, restart before each so HBM is empty
and the fault fires. **The metric is not the same on the two archs**, because the
arithmetic above says the win is not the same thing:

**V100 — turn TTFT.** The tier wins on the clock, so the clock is the test. At
`n* ≈ 73` the arms should be within noise; at 30,000 fetch should win by more than an
order of magnitude (derived **23.3x**, and I would accept anything above 5x as
confirming the mechanism rather than the exact figure — the derivation rests on a fit
extrapolated past the lengths that produced it).

**H20 — GPU seconds freed.** A 1.06x wall-clock win at 30k is inside the noise of a
contended box, so a turn-time comparison cannot accept or reject this. The claim
worth testing is the overlap one: **other sessions' decode and prefill throughput
while a fetch is in flight**, since the fetch occupies a reader thread and a copy
stream rather than the prefill kernels. The wall-clock arm still runs, as a
no-regression check only — reject if it goes *backwards*, do not accept it for
going forwards.

Required on both: `ssd_hits > 0` asserted per arm (a 0-hit arm measures only cost —
that is exactly what the 132-row 12-session run did), `compiles == 0` in every cell,
and the fetch's deadline drops counted. Reject on the V100 if fetch loses at 30k; on
the H20 if any decode row stalls by more than one tick.

Open: nothing blocking. `R` on the H20 is a seeded baseline from a contended box, so
the 14% margin could move; it cannot move enough to change which arch needs the
overlap argument.

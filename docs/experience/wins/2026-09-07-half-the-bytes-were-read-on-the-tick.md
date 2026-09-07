# Half the bytes a hit reads were read on the tick — H20 card 1, 2026-09-07

> Status: Shipped

## Context

The SSD tier's fault-in was moved off the tick in [the fetch runs off the
tick](2026-09-07-the-ssd-fetch-runs-off-the-tick.md): `submit` queues a prefetch, a
reader thread does the `torch.load`, and the tick copies from a blob that is already
in host memory.

That covered one of the two files. An entry is a `.kv` and a `.st`, and only the
`.kv` was prefetched. `load_state` read the `.st` on the **calling** thread at lookup
time — 157 MiB against the `.kv`'s 178 on a 2729-token entry, measured at **151 ms of
a 1.260 s faulted arm**. Neither `fetch_ms` nor `fetch_bytes` saw it, so no counter
showed the tick still doing disk I/O.

## What Worked

`_fetch_loop` reads both planes and parks them together; `take` returns the pair;
`_fault_in` hands the snapshot to `load_state` instead of letting it read. The
caller-side timer is deleted rather than kept, because on a hit there is nothing left
for it to time. `/health` still publishes `ssd_state_load_ms` and `ssd_state_loads`, now
always 0 — the keys are additive-only, so a reader that watches them keeps working and
reads the zero as "the lookup path no longer reads".

`B` is now the rate over everything a hit reads. That fixes an error the estimator
had from the start: `n* = (S/B) / (1/R − k/B)` divides the snapshot `S` by a rate
measured **without** `S` in it. Both planes come off one device through one reader
thread, so one rate is the whole model — two rates would add an operand the physics
does not have.

The test counts `torch.load` by thread rather than timing anything. Wall clock cannot
distinguish a read that moved from a read that got faster, and here it did not need to:
the warm arm did not move at all (see Results), so a timing gate would have called this
change a no-op.

```
both     ssd_hits 1   tick-side torch.load 0   reads on Thread-4 (_fetch_loop)
main     ssd_hits 1   tick-side torch.load 1   <- the .st, on the caller
```

## Rule

A subsystem with two files has two read paths, and moving one off the hot path proves
nothing about the other. The counter that would have shown the second one was absent
by construction — `fetch_ms` wrapped the prefetch, so work that never went through
the prefetch could not appear in it.

An instrument that measures only what it was pointed at will report success for the
half it can see.

## Results

**The warm ratio did not move, and that is the result.** Four runs on card 1 after the
change: **2.063 / 1.992 / 2.003 / 1.913**, mean 1.993. Five runs before it: 2.041 mean,
range 1.939–2.061. The ranges overlap, so nothing separates them.

The counters say why. Pre-change, instrumented: `fetch_ms` 122 + `state_load_ms` 151 =
**273 ms**. Post-change: one `fetch_ms` of **259 ms** mean (264 / 236 / 258 / 280), with
`state_load_ms` deleted. The bytes changed thread, they did not get cheaper — the
faulted arm is **1.262 s** mean against **1.260 s** before. A process restart empties
HBM and leaves the host page cache alone, so the `.st` the tick was reading came from
page cache, not from the device.

The device arm is where it should show. The bench composes the reboot/evicted case at a
measured **182.6 MiB/s**, where the same 157 MiB `.st` is a **0.86 s** device read — that
one is now off the tick, and this bench cannot measure it because its faulted arm is
page-cache warm.

What did change is what B prices: `fetch_mib_s` now covers 321.6 MiB per hit at
**1236 MiB/s** mean (1210.7 / 1354.3 / 1238.9 / 1141.5), against the `.kv` alone before.
`break_even_tokens` stays 8–9, because S and B grew by roughly the same factor; the
estimator got correct, its answer here did not move.

| date | commit | machine | target | model | arm | wall s | ratio |
|---|---|---|---|---|---|---:|---:|
| 2026-09-07 | 57a0e44+both-planes | H20 card 1 | cuda | Qwen3.8-27B NVFP4 | cold | 2.568 | — |
| 2026-09-07 | 57a0e44+both-planes | H20 card 1 | cuda | Qwen3.8-27B NVFP4 | faulted | 1.262 | 1.993x |
| 2026-09-07 | 57a0e44+both-planes | H20 card 1 | cuda | Qwen3.8-27B NVFP4 | control | 2.515 | 1.021x over cold |

Means of four runs; 2729-token prompt, 2720-token servable entry, `ssd_hits` 1,
`ssd_tick_loads` 0, `ssd_fetch_waits` 0 every run; `control_over_cold` inside the
bench's own 0.85–1.15 band.

Raw artifacts: card 1, `scripts/bench_ssd_restart.py --tokens 3000`, logs
`/work/row64a.log` and `/work/row64r{2,3,4}.log`; the pre-change run carrying both
timers is `/work/stinstr.log`.

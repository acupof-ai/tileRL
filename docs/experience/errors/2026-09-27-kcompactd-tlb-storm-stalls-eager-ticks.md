# Eager-refresh tick stalls were a kcompactd TLB-shootdown storm on our pinned memory — V100, 2026-09-27

Production sm70 (90c8a8b1, ThinkingCap-orig, sparse window 1024, refresh R32,
depth-1 draft, decode graph on, 4 slots, 8 GiB host cold tier): in a 2-turn
30k-character WebSocket chat (thinking on, ~18k prompt tokens/turn), turn 2's
answer showed periodic ~1.4 s frame gaps (recorded: reasoning frames ~29/61
each 1463 ms; one answer frame 1405 ms; turn 1 median gap 41 ms, turn 2 median
106 ms). It followed the #843 verify-Mb bucketing fix and was first suspected
to be a remaining JIT / cold-page / attention-kernel cost. It was the
**kernel's proactive memory compaction (`kcompactd`) repeatedly trying to
migrate our pinned (page-locked) memory: migration fails, the kernel still
unmaps and restores the PTEs, and each pass shoots down the TLB on the CPU
running the engine thread.**

## Context

The stall is not per-request: it arrives in clusters of 4-5 slow eager verify
ticks, roughly every 190 engine ticks, and is independent of which prompt turn
it is. `TILERL_STEP_TIMING=1` shows every GPU op in those ticks uniformly ~10x
slower (qkv/o linear GEMMs included, not just attention), with
`sparse_finalize` 2-15 ms, `offers_pages` mostly 0, `ssd_mmap=0`,
`d_malloc=0`, `free` constant at 1012 MiB, and `why=gpu_drain`. Wall time
equalled the engine thread's CPU time — C code slowed with Python, so this was
not a GIL/launch issue.

## Root Cause

`vm.compaction_proactiveness` defaults to 20. The kernel's `kcompactd` walks
memory looking for pages to migrate; the server holds a large set of
**pinned/page-locked** allocations (TileLang/pinned KV blobs and graph
buffers). Migration of a pinned page is refused, but the attempt still does an
unmap/remap cycle that sends an inter-processor interrupt (TLB shootdown) to
every CPU with a matching mapping — including the CPU running our single engine
thread. During a storm the engine core is repeatedly interrupted and its TLB is
repeatedly torn down, so the same in-order GPU work completes ~5x slower on the
wall clock and lands on the eager refresh ticks as 1.2-1.5 s gaps.

Measured (V100 host):

- During a storm the whole machine takes ~75,000 TLB shootdowns per 0.2 s; at
  rest it is 0. Storms arrive about every 40 s and last 3-5 s.
- `top` shows `kcompactd0` at 99.9% during a storm.
- `/proc/vmstat`: cumulative `pgmigrate_fail` ≈ 3.7 billion,
  `compact_success` ≈ 24 — almost every migration attempt fails.

It is not a GPU-side cause: external tenant contention, clock throttling,
compute steal, cgroup limits, page faults, Python GC, TileLang JIT, GIL
contention, trace hooks, vCPU scheduling, and L1/LLC/DRAM contention were each
checked with aligned probes and ruled out. There is also no in-repo side stream
that competes for SMs: all serving GPU work is issued in order on the default
stream by one engine thread (the only non-default streams are one-shot
decode-graph capture warmups that are fully `wait_stream`-synchronized).

## Fix

Disable proactive compaction at runtime on the serving host:

```
sudo sysctl vm.compaction_proactiveness=0
```

Controlled on/off comparison on the same V100, two 30k two-turn runs each:

- `0`: TLB-shootdown storm in 0 / 802 samples; max frame gap across both runs
  **223 ms**.
- `20` (default): storm in 72 / 421 samples; the ~1.4 s stalls reproduce.

The setting is persistent on the V100 host at
`/etc/sysctl.d/90-tilerl-no-proactive-compaction.conf`
(`vm.compaction_proactiveness=0`, per ckl 2026-09-27), so it survives reboot;
the runtime sysctl above only changes the live value. The engine code needs no
change.

## Rule

When every GPU op in a tick slows by the same factor (wall ≈ thread CPU time,
Python and C together), with unchanged geometry and no host/JIT/migration
signal, suspect a **kernel-level interruption of the CPU driving the GPU**, not
the kernels or the data path. Pinned-memory-heavy GPU servers are magnets for
`kcompactd` migration attempts; check `vm.compaction_proactiveness`, watch
`/proc/interrupts` TLB-shootdown counters and `/proc/vmstat` `pgmigrate_fail`
at sub-second cadence, and confirm with a 0/20 sysctl A/B. A CUDA `nvidia-smi`
view cannot see this — the contention is host CPU/TLB, not device SM.

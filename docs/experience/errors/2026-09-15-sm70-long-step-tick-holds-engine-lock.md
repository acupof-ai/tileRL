# sm70 step tick occasionally holds engine._lock for 1–5.6 s — 2026-09-15

**Status:** open — pending investigation, not fixed by the SSE cancel follow-up.
**Arch:** V100 sm70, dense+d1 (captured graph path)
**Discovered:** ops device verification of the F4 mid-stream disconnect fix
(72df5351, 36 reps); see
[2026-09-15-sse-midstream-disconnect-never-cancelled-the-row.md](2026-09-15-sse-midstream-disconnect-never-cancelled-the-row.md).

## Observed

A minority (~12%) of step ticks hold `engine._lock` for the whole forward,
1–5.6 s instead of the usual tens of ms. The window correlates with free VRAM
near **~350 MiB** on the dense+d1 path. A `stream_or_cancel` disconnect landing
inside one of those ticks parked behind the lock (that caller was fixed by
moving `engine.cancel` off the event loop); the long tick itself is untouched.

## Ruled out (so far)

- No JIT activity in the window (kernels were warm).
- Not eager fallback (captured graph path throughout).
- Not a sparse tick or a prefix cold/hot demote (dense+d1 only).

## Suspected, not verified

The CUDA allocator stalling on a nearly-full device is the leading suspicion
at ~350 MiB free, but there is no timing evidence yet.

## What the investigation needs

In-engine segmented timing on sm70 dense+d1: separate `_lock` acquisition
wait from the critical-section phases inside one long tick (plan, forward
launch, synchronize/allocator, commit), logged only when the tick crosses a
threshold (e.g. >500 ms), with `torch.cuda.memory_stats()` allocation/segment
counters at the same points. That distinguishes allocator stalls from a long
kernel or a plan/serialization section without running a heavy profiler on
the 27B server.

# A dead CUDA context is not an exception type, and the loop kept serving — V100 sm70, 2026-09-25

> Status: open — the fatal-exit fix is in this PR and not yet merged; the device
> confirmation (process exits 11, supervisor restarts it) is pending-remote. The
> **crash** this exposes is a separate open defect; the OPEN.md row names both.

## What happens

Production (V100 sm70, `9b8676a3`, ThinkingCap-orig, W1024/R32, narrow draft
query) raised `CUDA error: an illegal memory access was encountered` during a
37.6k-token request near its end (draft-window `first=2254`). The serving process
**kept running**: `/health` answered `status:ok` with `running=1` while the CUDA
context was already dead, so every subsequent request could only fail.

The trace shows both halves. The forward raised, then the same error surfaced
again from an unrelated call (`torch.cuda.mem_get_info`) — the signature of a
poisoned context, not of one bad kernel invocation:

```
engine.py:1300  step            -> self._run_forward(...)
engine.py:2067                  -> self._run_sparse_decode_graph(...)
sparse_runtime.py:846           -> ctx.verify(reqs, chains, logits, g.hidden)
engine.py:2641  _verify         -> self._sample_batch(flat)
engine.py:2689                  -> self._backend.sample_batch(...)
reference.py:1718               -> idx = torch.tensor(hot, device=dev)
RuntimeError: CUDA error: an illegal memory access was encountered
--- and then, from the loop's next tick ---
engine.py:1730  _device_free_limit -> torch.cuda.mem_get_info(...)
RuntimeError: CUDA error: an illegal memory access was encountered
```

The stack lands in `torch.tensor(hot, ...)` on the **host to device copy**, which
is where an asynchronously-reported error surfaces; the offending kernel is not
identified by this stack. That is what item 2 of the P0 is for.

## Why it kept serving

Two handlers, both doing the right thing for their own case:

- `Engine.step()` catches the forward's exception, `_finish`es every running row,
  and re-raises.
- `Engine._loop()` catches that with `except Exception: traceback.print_exc()` —
  the deliberate "log-and-continue" that stops a crashed daemon from hanging the
  server.

`FatalDeviceError` → `fatal_device_exit()` (`os._exit(11)` + a marker the
supervisor greps) already existed, but only `torch.cuda.OutOfMemoryError` was
routed to it. An illegal memory access is not an OOM, so it fell through to
log-and-continue.

## The trap: the obvious guard is vacuous

The natural fix is `except torch.cuda.CudaError`. **It would never fire on this
crash.** Measured on the card's own torch (`2.5.1+cu121`; identical shape on
`2.13`):

| | class | message |
|---|---|---|
| `torch.cuda.check_error(1)` | `torch.cuda.CudaError` | `invalid argument (1)` — **numeric suffix** |
| the crash above | plain `RuntimeError` | `CUDA error: an illegal memory access was encountered` — **no suffix** |

The second form is the c10 path, which raises a `RuntimeError`; `torch.cuda.CudaError`
is raised only by the cudart wrapper. They are also not related by inheritance:
`OutOfMemoryError` and `CudaError` are **siblings under `RuntimeError`**, so the
existing OOM branch does not catch it either. A guard written from the type name
would have shipped, passed its CPU test, and never fired on the real crash — an
instrument blind to its own failure.

## Fix

Ask the context, not the exception. `Backend.device_alive()` runs a
`torch.cuda.synchronize()` and returns False if anything raises; True on every
non-cuda target. `_loop()`'s broad handler calls it and routes to
`fatal_device_exit()` when it reads dead; otherwise the old log-and-continue is
unchanged. The OOM branch is untouched.

This covers the c10 form and the `CudaError` form with one predicate, needs no
message matching, and cannot misfire on an ordinary per-request error — a live
context is exactly what distinguishes them.

Supervisor side: the bounded loop already existed (`scripts/serve_hybrid_v100.sh`
— `MAX_RESTARTS`, a crash-burst fuse, a liveness guard) but its argv was baked in
for a different arm. It now takes `SERVE_ARGS`, and
`scripts/run_serve_v100_prod_supervised.sh` supplies the production arm, so the
restart machinery has one copy instead of one per arm.

## Gates

`tests/test_server.py`:

- `test_a_dead_cuda_context_is_fatal_even_for_a_survivable_exception` — drives the
  **real loop thread** with a forward raising the crash's exact class (plain
  `RuntimeError`) and the probe stubbed dead: asserts the exit seam fires once,
  the engine records `_fatal`, and `liveness()` reads dead. The negative control
  is the same exception with the probe stubbed **alive**: the seam must not fire.
- `test_device_alive_is_true_off_cuda_and_false_when_the_context_raises` — the
  predicate itself: True on a non-cuda target, and False (not a raise) when the
  synchronize raises, since the caller is already inside an `except` block.

**Negative control run:** disabling the probe call (`if False and not ...`) turns
the first gate red on its own assertion. Device-side confirmation that the
process exits 11 and the supervisor restarts it is **pending-remote**, to be
taken with the root-cause run.

**One test bug worth recording.** The first version waited with `if calls or not
alive: break`; with `alive=False` that breaks on the first iteration, before the
loop thread has run, and the gate then reports "0 exit calls" against a fix that
works. Wait on the thing you are measuring, not on a condition that is already
true.

## Still open

The crash itself. What is fixed is that a dead context can no longer be served
through. Root cause is a separate investigation: `CUDA_LAUNCH_BLOCKING=1` on the
same 37.6k prompt, with the suspected area being the own-window pages near the end
of generation, the `_select_device` gather (#824), and `own_w = WINDOW + 1` at
W1024.

## Related

- [an sm70 paged-attention launch hang wedges the engine](2026-09-15-sm70-paged-attention-launch-hang-wedges-engine.md)
  — the other way a tick can leave the engine unable to serve while the process
  stays up; that one hangs in a kernel, this one returns an error into a handler
  that keeps going.
